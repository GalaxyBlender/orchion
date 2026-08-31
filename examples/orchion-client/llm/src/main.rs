use orchion_client::llm::{
    ChatCompletionRequest, ChatMessage, ResponsesEvent, ResponsesInput, ResponsesRequest,
};
use orchion_client::{Client, ClientConfig};
use std::error::Error;
use std::io;

const USAGE: &str = "usage: client-llm <base-url> <model> <prompt>";

#[tokio::main]
async fn main() -> Result<(), Box<dyn Error>> {
    let args = required_args(3)?;
    let client = make_client(&args[0])?;

    let chat = client
        .llm()
        .create_chat_completion(ChatCompletionRequest::new(
            &args[1],
            vec![ChatMessage::user(&args[2])],
        ))
        .await?;
    println!("chat completion:");
    for choice in chat.choices {
        println!("{}", choice.message.content);
    }

    println!("Responses stream:");
    let mut stream = client
        .llm()
        .stream_response(ResponsesRequest::new(
            &args[1],
            ResponsesInput::text(&args[2]),
        ))
        .await?;
    loop {
        match stream.next_event().await? {
            Some(ResponsesEvent::Created { response, .. }) => {
                println!("created: {}", response.id);
            }
            Some(ResponsesEvent::InProgress { .. }) => println!("in progress"),
            Some(ResponsesEvent::OutputTextDelta { delta, .. }) => print!("{delta}"),
            Some(ResponsesEvent::Completed { response, .. }) => {
                println!("\ncompleted: {}", response.id);
                break;
            }
            Some(ResponsesEvent::Incomplete { response, .. }) => {
                return Err(io::Error::other(format!(
                    "Responses request ended incomplete: {:?}",
                    response.incomplete_details
                ))
                .into());
            }
            Some(_) => {}
            None => {
                return Err(
                    io::Error::other("Responses stream ended without a terminal event").into(),
                );
            }
        }
    }
    Ok(())
}

fn required_args(count: usize) -> Result<Vec<String>, io::Error> {
    let args: Vec<_> = std::env::args().skip(1).collect();
    if args.len() == count {
        Ok(args)
    } else {
        Err(io::Error::new(io::ErrorKind::InvalidInput, USAGE))
    }
}

fn make_client(base_url: &str) -> Result<Client, Box<dyn Error>> {
    let mut config = ClientConfig::new(base_url)?;
    match std::env::var("ORCHION_API_KEY") {
        Ok(api_key) if !api_key.is_empty() => config = config.with_api_key(api_key),
        Ok(_) | Err(std::env::VarError::NotPresent) => {}
        Err(error) => return Err(error.into()),
    }
    Ok(Client::from_config(config)?)
}
