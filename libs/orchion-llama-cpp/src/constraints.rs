use crate::contract::{Error, OutputConstraint};

const MAX_SCHEMA_BYTES: usize = 64 * 1024;
const MAX_SCHEMA_DEPTH: usize = 32;
const MAX_SCHEMA_NODES: usize = 4096;
const MAX_GRAMMAR_BYTES: usize = 256 * 1024;
const ALLOWED_SCHEMA_KEYWORDS: &[&str] = &[
    "type",
    "properties",
    "required",
    "additionalProperties",
    "items",
    "enum",
    "const",
    "minimum",
    "maximum",
    "minLength",
    "maxLength",
    "minItems",
    "maxItems",
    "description",
    "title",
];

pub(crate) fn grammar_for_constraint(
    constraint: &OutputConstraint,
) -> Result<Option<String>, Error> {
    let grammar = match constraint {
        OutputConstraint::Text => return Ok(None),
        OutputConstraint::JsonObject => {
            llama_cpp_2::json_schema_to_grammar(r#"{"type":"object","additionalProperties":true}"#)
                .map_err(|error| Error::InvalidConfig(format!("JSON object constraint: {error}")))?
        }
        OutputConstraint::JsonSchema(schema) => {
            validate_strict_schema(schema)?;
            let json = serde_json::to_string(schema)
                .map_err(|error| Error::InvalidConfig(format!("JSON schema: {error}")))?;
            llama_cpp_2::json_schema_to_grammar(&json)
                .map_err(|error| Error::InvalidConfig(format!("JSON schema constraint: {error}")))?
        }
        OutputConstraint::Grammar(grammar) => {
            if grammar.is_empty() || grammar.len() > MAX_GRAMMAR_BYTES {
                return Err(Error::InvalidConfig(format!(
                    "grammar must contain 1..={MAX_GRAMMAR_BYTES} bytes"
                )));
            }
            if grammar.contains('\0') || !grammar.contains("root ::=") {
                return Err(Error::InvalidConfig(
                    "grammar must be NUL-free and define `root`".to_string(),
                ));
            }
            grammar.clone()
        }
    };
    if grammar.len() > MAX_GRAMMAR_BYTES {
        return Err(Error::InvalidConfig(format!(
            "compiled grammar exceeds {MAX_GRAMMAR_BYTES} bytes"
        )));
    }
    Ok(Some(grammar))
}

pub fn validate_strict_schema(schema: &serde_json::Value) -> Result<(), Error> {
    let serialized = serde_json::to_vec(schema)
        .map_err(|error| Error::InvalidConfig(format!("JSON schema: {error}")))?;
    if serialized.len() > MAX_SCHEMA_BYTES {
        return invalid_schema(format!("schema exceeds {MAX_SCHEMA_BYTES} bytes"));
    }
    let root = schema
        .as_object()
        .ok_or_else(|| schema_error("root must be an object"))?;
    if root.get("type").and_then(serde_json::Value::as_str) != Some("object") {
        return invalid_schema("root type must be `object`");
    }
    let mut nodes = 0;
    validate_schema_node(schema, 0, &mut nodes)
}

fn validate_schema_node(
    value: &serde_json::Value,
    depth: usize,
    nodes: &mut usize,
) -> Result<(), Error> {
    if depth > MAX_SCHEMA_DEPTH {
        return invalid_schema(format!("schema exceeds depth {MAX_SCHEMA_DEPTH}"));
    }
    *nodes += 1;
    if *nodes > MAX_SCHEMA_NODES {
        return invalid_schema(format!("schema exceeds {MAX_SCHEMA_NODES} nodes"));
    }
    let object = value
        .as_object()
        .ok_or_else(|| schema_error("every schema node must be an object"))?;
    if let Some(keyword) = object
        .keys()
        .find(|key| !ALLOWED_SCHEMA_KEYWORDS.contains(&key.as_str()))
    {
        return invalid_schema(format!("unsupported keyword `{keyword}`"));
    }
    let schema_type = object
        .get("type")
        .and_then(serde_json::Value::as_str)
        .ok_or_else(|| schema_error("every schema node must declare one string `type`"))?;
    if !matches!(
        schema_type,
        "object" | "array" | "string" | "number" | "integer" | "boolean" | "null"
    ) {
        return invalid_schema(format!("unsupported type `{schema_type}`"));
    }
    if let Some(values) = object.get("enum") {
        let values = values
            .as_array()
            .ok_or_else(|| schema_error("`enum` must be an array"))?;
        if values.is_empty() || values.len() > 256 {
            return invalid_schema("`enum` must contain 1..=256 values");
        }
        if values
            .iter()
            .any(|value| !value_matches_type(value, schema_type))
        {
            return invalid_schema("every `enum` value must match the declared type");
        }
    }
    if object
        .get("const")
        .is_some_and(|value| !value_matches_type(value, schema_type))
    {
        return invalid_schema("`const` must match the declared type");
    }
    for annotation in ["description", "title"] {
        if object
            .get(annotation)
            .is_some_and(|value| !value.is_string())
        {
            return invalid_schema(format!("`{annotation}` must be a string"));
        }
    }
    validate_numeric_bounds(object, schema_type)?;
    validate_size_bounds(object, schema_type)?;
    if (object.contains_key("minimum") || object.contains_key("maximum"))
        && !matches!(schema_type, "number" | "integer")
    {
        return invalid_schema("`minimum` and `maximum` require a numeric type");
    }
    if (object.contains_key("minLength") || object.contains_key("maxLength"))
        && schema_type != "string"
    {
        return invalid_schema("`minLength` and `maxLength` require type `string`");
    }
    if (object.contains_key("minItems") || object.contains_key("maxItems"))
        && schema_type != "array"
    {
        return invalid_schema("`minItems` and `maxItems` require type `array`");
    }
    match schema_type {
        "object" => validate_object_schema(object, depth, nodes),
        "array" => {
            let items = object
                .get("items")
                .ok_or_else(|| schema_error("array schemas must declare `items`"))?;
            validate_schema_node(items, depth + 1, nodes)
        }
        _ => {
            if object.contains_key("properties")
                || object.contains_key("required")
                || object.contains_key("additionalProperties")
                || object.contains_key("items")
            {
                return invalid_schema(format!(
                    "type `{schema_type}` contains object/array-only keywords"
                ));
            }
            Ok(())
        }
    }
}

fn value_matches_type(value: &serde_json::Value, schema_type: &str) -> bool {
    match schema_type {
        "object" => value.is_object(),
        "array" => value.is_array(),
        "string" => value.is_string(),
        "number" => value.is_number(),
        "integer" => value.as_i64().is_some() || value.as_u64().is_some(),
        "boolean" => value.is_boolean(),
        "null" => value.is_null(),
        _ => false,
    }
}

fn validate_numeric_bounds(
    object: &serde_json::Map<String, serde_json::Value>,
    schema_type: &str,
) -> Result<(), Error> {
    let read = |keyword: &'static str| -> Result<Option<f64>, Error> {
        object
            .get(keyword)
            .map(|value| {
                value
                    .as_f64()
                    .filter(|value| value.is_finite())
                    .ok_or_else(|| schema_error(format!("`{keyword}` must be a finite number")))
            })
            .transpose()
    };
    let minimum = read("minimum")?;
    let maximum = read("maximum")?;
    if matches!(schema_type, "integer")
        && [minimum, maximum]
            .into_iter()
            .flatten()
            .any(|value| value.fract() != 0.0)
    {
        return invalid_schema("integer bounds must be integral");
    }
    if minimum.zip(maximum).is_some_and(|(min, max)| min > max) {
        return invalid_schema("`minimum` must not exceed `maximum`");
    }
    Ok(())
}

fn validate_size_bounds(
    object: &serde_json::Map<String, serde_json::Value>,
    schema_type: &str,
) -> Result<(), Error> {
    let pair = match schema_type {
        "string" => ("minLength", "maxLength"),
        "array" => ("minItems", "maxItems"),
        _ => return Ok(()),
    };
    let read = |keyword: &'static str| -> Result<Option<u64>, Error> {
        object
            .get(keyword)
            .map(|value| {
                value
                    .as_u64()
                    .filter(|value| *value <= 1_000_000)
                    .ok_or_else(|| {
                        schema_error(format!("`{keyword}` must be an integer in 0..=1000000"))
                    })
            })
            .transpose()
    };
    let minimum = read(pair.0)?;
    let maximum = read(pair.1)?;
    if minimum.zip(maximum).is_some_and(|(min, max)| min > max) {
        return invalid_schema(format!("`{}` must not exceed `{}`", pair.0, pair.1));
    }
    Ok(())
}

fn validate_object_schema(
    object: &serde_json::Map<String, serde_json::Value>,
    depth: usize,
    nodes: &mut usize,
) -> Result<(), Error> {
    if object.get("additionalProperties") != Some(&serde_json::Value::Bool(false)) {
        return invalid_schema("object schemas must set `additionalProperties` to false");
    }
    let properties = object
        .get("properties")
        .and_then(serde_json::Value::as_object)
        .ok_or_else(|| schema_error("object schemas must declare `properties`"))?;
    let required = object
        .get("required")
        .and_then(serde_json::Value::as_array)
        .ok_or_else(|| schema_error("object schemas must declare `required`"))?;
    if required.len() != properties.len()
        || required.iter().any(|name| {
            name.as_str()
                .is_none_or(|name| !properties.contains_key(name))
        })
    {
        return invalid_schema("strict object schemas must require every property exactly once");
    }
    let mut unique = std::collections::BTreeSet::new();
    if required
        .iter()
        .filter_map(serde_json::Value::as_str)
        .any(|name| !unique.insert(name))
    {
        return invalid_schema("`required` contains duplicate properties");
    }
    for child in properties.values() {
        validate_schema_node(child, depth + 1, nodes)?;
    }
    Ok(())
}

fn invalid_schema<T>(detail: impl Into<String>) -> Result<T, Error> {
    Err(schema_error(detail))
}

fn schema_error(detail: impl Into<String>) -> Error {
    Error::InvalidConfig(format!("strict JSON schema: {}", detail.into()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn strict_schema_accepts_closed_required_subset_and_compiles() {
        let schema = serde_json::json!({
            "type": "object",
            "properties": {
                "name": {"type": "string", "maxLength": 20},
                "count": {"type": "integer", "minimum": 0}
            },
            "required": ["name", "count"],
            "additionalProperties": false
        });
        assert!(validate_strict_schema(&schema).is_ok());
        assert!(
            grammar_for_constraint(&OutputConstraint::JsonSchema(schema))
                .unwrap()
                .unwrap()
                .contains("root ::=")
        );
    }

    #[test]
    fn strict_schema_rejects_keywords_or_shapes_that_could_be_weakened() {
        for schema in [
            serde_json::json!({"type":"object","properties":{},"required":[],"additionalProperties":true}),
            serde_json::json!({"type":"object","properties":{"x":{"type":"string"}},"required":[],"additionalProperties":false}),
            serde_json::json!({"type":"object","properties":{},"required":[],"additionalProperties":false,"oneOf":[]}),
            serde_json::json!({"type":"object","properties":{"x":{"type":"string","minLength":4,"maxLength":2}},"required":["x"],"additionalProperties":false}),
            serde_json::json!({"type":"object","properties":{"x":{"type":"integer","minimum":0.5}},"required":["x"],"additionalProperties":false}),
            serde_json::json!({"type":"object","properties":{"x":{"type":"boolean","enum":["true"]}},"required":["x"],"additionalProperties":false}),
        ] {
            assert!(validate_strict_schema(&schema).is_err());
        }
    }
}
