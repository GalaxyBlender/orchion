//! Render model-generated tables without losing cells or trusting generated HTML.

use htmd::{HtmlToMarkdown, node::Node};
use std::fmt::Write;

struct Cell {
    text: String,
    header: bool,
    colspan: u32,
    rowspan: u32,
}

#[derive(Default)]
struct Table {
    caption: Option<String>,
    rows: Vec<Vec<Cell>>,
}

impl Table {
    fn from_node(node: &Node<'_>) -> Self {
        let mut table = Self::default();
        table.collect_rows(node);
        table
    }

    fn collect_rows(&mut self, node: &Node<'_>) {
        for child in &node.children() {
            match child.tag_name() {
                Some("caption") => self.caption = Some(cell_text(&child)),
                Some("thead" | "tbody" | "tfoot") => self.collect_rows(&child),
                Some("tr") => {
                    let cells = child
                        .children()
                        .iter()
                        .filter_map(|cell| {
                            let tag = cell.tag_name()?;
                            if !matches!(tag, "td" | "th") {
                                return None;
                            }
                            let span = |attribute: &str| {
                                cell.attr(attribute)
                                    .and_then(|value| value.parse::<u32>().ok())
                                    .filter(|value| *value > 0)
                                    .unwrap_or(1)
                            };
                            Some(Cell {
                                text: cell_text(&cell),
                                header: tag == "th",
                                colspan: span("colspan"),
                                rowspan: span("rowspan"),
                            })
                        })
                        .collect::<Vec<_>>();
                    if !cells.is_empty() {
                        self.rows.push(cells);
                    }
                }
                _ => {}
            }
        }
    }

    fn html(&self) -> String {
        let mut html = String::from("<table>");
        if let Some(caption) = &self.caption {
            write!(html, "<caption>{}</caption>", escape_html(caption))
                .expect("writing to a String cannot fail");
        }
        for row in &self.rows {
            html.push_str("<tr>");
            for cell in row {
                let tag = if cell.header { "th" } else { "td" };
                html.push('<');
                html.push_str(tag);
                for (name, span) in [("colspan", cell.colspan), ("rowspan", cell.rowspan)] {
                    if span > 1 {
                        write!(html, " {name}=\"{span}\"")
                            .expect("writing to a String cannot fail");
                    }
                }
                html.push('>');
                html.push_str(&escape_html(&cell.text).replace('\n', "<br>"));
                write!(html, "</{tag}>").expect("writing to a String cannot fail");
            }
            html.push_str("</tr>");
        }
        html.push_str("</table>");
        html
    }

    fn markdown(&self) -> String {
        // GFM cannot express spans. Keep content in the top-left cell and
        // reserve the covered positions, rather than returning an HTML table.
        // None is an unused slot; Some("") is a slot covered by a merged cell.
        let mut grid = vec![Vec::<Option<&str>>::new(); self.rows.len()];
        for (row_index, row) in self.rows.iter().enumerate() {
            let mut column = 0;
            for cell in row {
                // HTML bounds colspan at 1000. Model output is untrusted and
                // must not amplify a generated span into billions of slots.
                let colspan = usize::try_from(cell.colspan.min(1000))
                    .expect("bounded HTML colspan fits usize");
                while let Some(offset) = grid[row_index]
                    .iter()
                    .skip(column)
                    .take(colspan)
                    .position(Option::is_some)
                {
                    column += offset + 1;
                }
                let end = column + colspan;
                let rowspan =
                    usize::try_from(cell.rowspan).expect("u32 rowspan fits supported targets");
                for covered_row in grid.iter_mut().skip(row_index).take(rowspan) {
                    if covered_row.len() < end {
                        covered_row.resize(end, None);
                    }
                    covered_row[column..end].fill(Some(""));
                }
                grid[row_index][column] = Some(&cell.text);
                column = end;
            }
        }
        let columns = grid.iter().map(Vec::len).max().unwrap_or(0);
        let rows = grid
            .into_iter()
            .map(|mut row| {
                row.resize(columns, None);
                row.into_iter()
                    .map(|text| markdown_cell(text.unwrap_or_default()))
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        let widths = (0..columns)
            .map(|column| {
                rows.iter()
                    .map(|row| row[column].chars().count())
                    .max()
                    .unwrap_or(0)
            })
            .collect::<Vec<_>>();
        let mut markdown = String::new();
        if let Some(caption) = &self.caption {
            write!(markdown, "{}\n\n", markdown_cell(caption))
                .expect("writing to a String cannot fail");
        }
        for (index, row) in rows.iter().enumerate() {
            markdown.push('|');
            for (cell, width) in row.iter().zip(&widths) {
                write!(markdown, " {cell:width$} |", width = *width)
                    .expect("writing to a String cannot fail");
            }
            markdown.push('\n');
            if index == 0 {
                markdown.push('|');
                for width in &widths {
                    write!(markdown, " {} |", "-".repeat((*width).max(3)))
                        .expect("writing to a String cannot fail");
                }
                markdown.push('\n');
            }
        }
        markdown.trim_end_matches('\n').to_string()
    }

    fn text(&self) -> String {
        let mut lines = Vec::new();
        if let Some(caption) = &self.caption {
            lines.push(caption.split_whitespace().collect::<Vec<_>>().join(" "));
        }
        lines.extend(self.rows.iter().map(|row| {
            row.iter()
                .map(|cell| cell.text.split_whitespace().collect::<Vec<_>>().join(" "))
                .collect::<Vec<_>>()
                .join("\t")
        }));
        lines.join("\n")
    }
}

fn markdown_cell(text: &str) -> String {
    let text = text.split_whitespace().collect::<Vec<_>>().join(" ");
    let mut escaped = String::new();
    for character in text.chars() {
        match character {
            '&' => escaped.push_str("&amp;"),
            '<' => escaped.push_str("&lt;"),
            '>' => escaped.push_str("&gt;"),
            '|' => escaped.push_str("&#124;"),
            '\\' | '*' | '_' | '`' | '[' | ']' | '~' => {
                escaped.push('\\');
                escaped.push(character);
            }
            _ => escaped.push(character),
        }
    }
    escaped
}

fn cell_text(node: &Node<'_>) -> String {
    fn collect(node: &Node<'_>, text: &mut String) {
        if let Some(contents) = node.text() {
            text.push_str(&contents);
        }
        let tag = node.tag_name();
        let block = matches!(tag, Some("p" | "div" | "li"));
        match tag {
            Some("script" | "style" | "template") => return,
            Some("br") => text.push('\n'),
            _ if block && !text.ends_with('\n') => text.push('\n'),
            _ => {}
        }
        for child in &node.children() {
            collect(&child, text);
        }
        if block && !text.ends_with('\n') {
            text.push('\n');
        }
    }
    let mut text = String::new();
    collect(node, &mut text);
    text.trim().to_string()
}

fn unfence_table(input: &str) -> &str {
    let trimmed = input.trim();
    let body = trimmed
        .strip_prefix("```html\n")
        .or_else(|| trimmed.strip_prefix("```\n"))
        .and_then(|body| body.strip_suffix("\n```"));
    body.filter(|body| body.contains("<table")).unwrap_or(input)
}

pub(crate) fn table_to_html(input: &str) -> Option<String> {
    fn render(node: &Node<'_>, output: &mut String) {
        let mut block = false;
        if let Some(contents) = node.text() {
            output.push_str(&escape_html(&contents));
        }
        match node.tag_name() {
            Some("table") => {
                output.push_str(&Table::from_node(node).html());
                return;
            }
            Some("head" | "script" | "style" | "template") => return,
            Some("br") => output.push_str("<br>"),
            Some("div") => output.push('\n'),
            Some("p" | "h1" | "h2" | "h3" | "li") => {
                block = true;
                output.push_str("<p>");
            }
            _ => {}
        }
        for child in &node.children() {
            render(&child, output);
        }
        if block {
            output.push_str("</p>");
        }
    }
    let tree = HtmlToMarkdown::new()
        .html_to_tree(unfence_table(input))
        .expect("parsing an in-memory HTML string cannot fail");
    let root = tree.root();
    let node = root.find(|node| node.tag_name() == Some("table"))?;
    let table = Table::from_node(&node);
    if table.rows.is_empty() {
        return None;
    }
    let mut html = String::new();
    render(&root, &mut html);
    Some(html.trim().to_string())
}

/// Only model-generated table markup is decoded; ordinary OCR text stays literal.
pub(crate) fn table_to_text(input: &str) -> Option<String> {
    fn line_break(output: &mut String) {
        if !output.is_empty() && !output.ends_with('\n') {
            output.push('\n');
        }
    }
    fn render(node: &Node<'_>, output: &mut String) {
        let mut block = false;
        if let Some(contents) = node.text() {
            output.push_str(&contents);
        }
        match node.tag_name() {
            Some("table") => {
                line_break(output);
                output.push_str(&Table::from_node(node).text());
                line_break(output);
                return;
            }
            Some("head" | "script" | "style" | "template") => return,
            Some("br") => output.push('\n'),
            Some("p" | "div" | "h1" | "h2" | "h3" | "li") => {
                block = true;
                line_break(output);
            }
            _ => {}
        }
        for child in &node.children() {
            render(&child, output);
        }
        if block {
            line_break(output);
        }
    }
    let tree = HtmlToMarkdown::new()
        .html_to_tree(unfence_table(input))
        .expect("parsing an in-memory HTML string cannot fail");
    let root = tree.root();
    root.find(|node| node.tag_name() == Some("table"))?;
    let mut text = String::new();
    render(&root, &mut text);
    // Keep tabs at either end: they represent empty first/last cells.
    Some(text.trim_matches('\n').to_string())
}

pub(crate) fn html_tables_to_markdown(input: &str) -> String {
    fn render(node: &Node<'_>, output: &mut String) {
        let mut block = false;
        if let Some(contents) = node.text() {
            output.push_str(&escape_html(&contents));
        }
        match node.tag_name() {
            Some("table") => {
                output.push_str("\n\n");
                output.push_str(&Table::from_node(node).markdown());
                output.push_str("\n\n");
                return;
            }
            Some("head" | "script" | "style" | "template") => return,
            Some("br") => output.push('\n'),
            Some("p" | "div") => {
                block = true;
                output.push_str("\n\n");
            }
            _ => {}
        }
        for child in &node.children() {
            render(&child, output);
        }
        if block {
            output.push_str("\n\n");
        }
    }

    // Document parsers already return Markdown outside the HTML tables. Running
    // that text through an HTML-to-Markdown converter again escapes headings and
    // emphasis and collapses paragraph/list newlines.
    let tree = HtmlToMarkdown::new()
        .html_to_tree(unfence_table(input))
        .expect("parsing an in-memory HTML string cannot fail");
    let mut markdown = String::new();
    render(&tree.root(), &mut markdown);
    markdown.trim_matches('\n').to_string()
}

pub(crate) fn escape_html(text: &str) -> String {
    text.chars().fold(String::new(), |mut escaped, character| {
        match character {
            '&' => escaped.push_str("&amp;"),
            '<' => escaped.push_str("&lt;"),
            '>' => escaped.push_str("&gt;"),
            _ => escaped.push(character),
        }
        escaped
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn markdown_cells(markdown: &str) -> Vec<Vec<&str>> {
        markdown
            .lines()
            .filter(|line| line.starts_with('|'))
            .map(|line| {
                line.strip_prefix('|')
                    .unwrap()
                    .strip_suffix('|')
                    .unwrap()
                    .split('|')
                    .map(str::trim)
                    .collect()
            })
            .collect()
    }

    #[test]
    fn markdown_expands_merged_title_instead_of_returning_an_html_document() {
        // Same shape as the live response: a six-column title plus an empty
        // seventh cell, followed by seven-column data rows.
        let markdown = html_tables_to_markdown(
            "<table><tr><td colspan=6>Project summary</td><td></td></tr><tr><td>No.</td><td>Project</td><td>Qty</td><td>Price</td><td>Total</td><td>Notes</td><td></td></tr><tr><td>1</td><td>Keyboard</td><td>2</td><td>49.00</td><td>98.00</td><td></td><td></td></tr></table>",
        );
        assert!(!markdown.contains("<table"), "{markdown}");
        let rows = markdown_cells(&markdown);
        assert_eq!(rows.len(), 4);
        assert!(rows.iter().all(|row| row.len() == 7), "{markdown}");
        assert!(
            rows[1]
                .iter()
                .all(|cell| !cell.is_empty() && cell.chars().all(|character| character == '-')),
            "every column needs a Markdown delimiter: {markdown}"
        );
        assert_eq!(rows[0], ["Project summary", "", "", "", "", "", ""]);
        assert_eq!(
            rows[2],
            ["No.", "Project", "Qty", "Price", "Total", "Notes", ""]
        );
        assert_eq!(rows[3], ["1", "Keyboard", "2", "49.00", "98.00", "", ""]);
    }

    #[test]
    fn markdown_keeps_column_positions_under_rowspans_and_pads_ragged_rows() {
        let markdown = html_tables_to_markdown(
            "<table><tr><td colspan=3>Sales</td></tr><tr><td rowspan=2>A</td><td>B</td><td>C</td></tr><tr><td>D</td><td>E</td></tr><tr><td>F</td></tr></table>",
        );
        assert!(!markdown.contains("<table"), "{markdown}");
        let rows = markdown_cells(&markdown);
        assert_eq!(rows.len(), 5);
        assert_eq!(rows[0], ["Sales", "", ""]);
        assert_eq!(rows[2], ["A", "B", "C"]);
        assert_eq!(rows[3], ["", "D", "E"]);
        assert_eq!(rows[4], ["F", "", ""]);
    }

    #[test]
    fn markdown_keeps_valid_delimiters_for_completely_empty_columns() {
        let markdown = html_tables_to_markdown(
            "<table><tr><td></td><td></td></tr><tr><td></td><td></td></tr></table>",
        );
        assert_eq!(
            markdown_cells(&markdown),
            vec![vec!["", ""], vec!["---", "---"], vec!["", ""]]
        );
    }

    #[test]
    fn markdown_treats_recognized_inline_syntax_as_literal_cell_text() {
        let markdown = html_tables_to_markdown(
            "<table><tr><td>Name</td></tr><tr><td>*not emphasis* [not a link] _literal_ `code`</td></tr></table>",
        );
        assert!(
            markdown.contains(r"\*not emphasis\* \[not a link\] \_literal\_ \`code\`"),
            "{markdown}"
        );
    }

    #[test]
    fn markdown_spans_keep_literal_text_without_html_or_multiline_rows() {
        let markdown = html_tables_to_markdown(
            "<table><caption>Sales &amp; tax</caption><tr><td colspan=2>Title</td></tr><tr><td>A | B<br>C</td><td>&lt;literal&gt; &amp; value</td></tr></table>",
        );
        assert!(markdown.contains("Sales"), "{markdown}");
        assert!(!markdown.contains("<table"), "{markdown}");
        assert!(!markdown.contains("<br"), "{markdown}");
        assert!(markdown.contains("&lt;literal&gt;"), "{markdown}");
        assert!(
            markdown.contains("A &#124; B C") || markdown.contains(r"A \| B C"),
            "{markdown}"
        );
        assert_eq!(markdown_cells(&markdown).len(), 3);
    }

    #[test]
    fn markdown_bounds_generated_spans_without_inventing_extra_rows() {
        let markdown = html_tables_to_markdown(
            "<table><tr><td colspan=4294967295 rowspan=4294967295>A</td></tr><tr><td>B</td></tr></table>",
        );
        let rows = markdown_cells(&markdown);
        assert_eq!(rows.len(), 3);
        assert!(rows.iter().all(|row| row.len() == 1001));
        assert_eq!(rows[0][0], "A");
        assert_eq!(rows[2][1000], "B");
    }

    #[test]
    fn plain_table_text_preserves_rows_empty_cells_and_decodes_entities() {
        let text = table_to_text(
            "<html><body><table><caption>Sales &amp; tax</caption><tr><td></td><th>Name</th><th>Total</th><td></td></tr><tr><td></td><td>A &amp; B</td><td>98.00</td><td></td></tr></table></body></html>",
        ).unwrap();
        assert_eq!(text, "Sales & tax\n\tName\tTotal\t\n\tA & B\t98.00\t");
        assert!(!text.contains("<table"));
    }

    #[test]
    fn plain_table_text_handles_merged_and_multiline_cells_without_duplicate_content() {
        let text = table_to_text(
            "<table><tr><td colspan=2>Title</td></tr><tr><td rowspan=2>A<br>B</td><td>10</td></tr><tr><td><p>C</p><p>D</p>20<script>evil()</script></td></tr></table>",
        ).unwrap();
        assert_eq!(text, "Title\nA B\t10\nC D 20");
    }

    #[test]
    fn table_formats_keep_multiple_tables_and_surrounding_text_in_order() {
        let input = "<html><head><title>not document content</title></head><body><p>Before</p><table onclick=evil()><tr><td>A &amp; B</td><td>1</td></tr></table><p>Between</p><table><tr><td>C</td><td>2</td></tr></table><p>After</p><script>evil()</script></body></html>";
        assert_eq!(
            table_to_text(input).unwrap(),
            "Before\nA & B\t1\nBetween\nC\t2\nAfter"
        );
        for output in [
            table_to_html(input).unwrap(),
            html_tables_to_markdown(input),
        ] {
            assert!(!output.contains("evil()"), "{output}");
            assert!(!output.contains("not document content"), "{output}");
            let mut remaining = output.as_str();
            for expected in ["Before", "A &amp; B", "Between", "C", "After"] {
                // Markdown decodes ampersands while HTML escapes them.
                let expected = if !remaining.contains(expected) && expected == "A &amp; B" {
                    "A & B"
                } else {
                    expected
                };
                let position = remaining.find(expected).unwrap();
                remaining = &remaining[position + expected.len()..];
            }
        }
    }

    #[test]
    fn table_text_does_not_treat_ordinary_ocr_content_as_markup() {
        assert_eq!(
            table_to_text("Use <script> as a literal name; x < y & y > z"),
            None
        );
        assert_eq!(table_to_text("Name\tValue\nA\t1"), None);
    }

    #[test]
    fn truncated_table_text_keeps_cells() {
        assert_eq!(
            table_to_text("<table><tr><td>Keyboard</td><td>98.00"),
            Some("Keyboard\t98.00".to_string())
        );
    }

    #[test]
    fn generated_table_code_fences_are_not_returned_as_document_content() {
        for fence in ["```html", "```"] {
            let input = format!(
                "{fence}\n<html><body><table><tr><td>A</td><td>1</td></tr></table></body></html>\n```"
            );
            assert_eq!(table_to_text(&input).as_deref(), Some("A\t1"));
            assert_eq!(
                table_to_html(&input).as_deref(),
                Some("<table><tr><td>A</td><td>1</td></tr></table>")
            );
            assert!(!html_tables_to_markdown(&input).contains("```"));
        }
    }

    #[test]
    fn html_table_wrappers_do_not_create_invalid_paragraphs_around_tables() {
        let input = "<div><table><tr><td>A</td></tr></table></div>";
        assert_eq!(
            table_to_html(input).as_deref(),
            Some("<table><tr><td>A</td></tr></table>")
        );
    }

    #[test]
    fn table_html_preserves_empty_cells_and_rejects_executable_markup() {
        let html = table_to_html(
            "<table onclick=\"evil()\"><tr><td><img src=x onerror=evil()>A &amp; B<script>evil()</script></td><td></td></tr></table>",
        )
        .unwrap();
        assert_eq!(html, "<table><tr><td>A &amp; B</td><td></td></tr></table>");
    }

    #[test]
    fn truncated_model_output_keeps_recognized_cells() {
        let html = table_to_html("<table><tr><td>Keyboard</td><td>98.00").unwrap();
        assert_eq!(
            html,
            "<table><tr><td>Keyboard</td><td>98.00</td></tr></table>"
        );
    }

    #[test]
    fn markdown_escapes_literal_tags_and_preserves_empty_cells() {
        let markdown = html_tables_to_markdown(
            "<table><tr><td>Name</td><td>Value</td></tr><tr><td>&lt;script&gt;literal&lt;/script&gt;</td><td></td></tr></table>",
        );
        assert!(!markdown.contains("<script>"));
        assert!(markdown.contains("&lt;script&gt;"));
        assert_eq!(markdown.lines().last().unwrap().matches('|').count(), 3);
    }

    #[test]
    fn markdown_cell_pipes_do_not_create_extra_columns() {
        let markdown = html_tables_to_markdown(
            "<table><tr><td>Name</td><td>Value</td></tr><tr><td>A | B</td><td>1</td></tr></table>",
        );
        assert!(
            markdown.contains(r"A \| B") || markdown.contains("A &#124; B"),
            "{markdown}"
        );
        assert_eq!(markdown.lines().count(), 3);
    }

    #[test]
    fn markdown_keeps_separate_tables_in_document_order() {
        let markdown = html_tables_to_markdown(
            "<p>Before</p><table><tr><td>A</td></tr></table><p>Between</p><table><tr><td>B</td></tr></table><p>After</p>",
        );
        let mut remaining = markdown.as_str();
        for expected in ["Before", "| A |", "Between", "| B |", "After"] {
            let position = remaining.find(expected).unwrap();
            remaining = &remaining[position + expected.len()..];
        }
    }

    #[test]
    fn table_conversion_preserves_surrounding_document_markdown() {
        let markdown = html_tables_to_markdown(
            "# Invoice\n\nTotal **before tax**\n\n<table><tr><td>A</td><td>1</td></tr></table>\n\n## Notes\n\n- paid\n- shipped",
        );
        assert!(
            markdown.starts_with("# Invoice\n\nTotal **before tax**\n\n"),
            "{markdown}"
        );
        assert!(
            markdown.ends_with("## Notes\n\n- paid\n- shipped"),
            "{markdown}"
        );
    }
}
