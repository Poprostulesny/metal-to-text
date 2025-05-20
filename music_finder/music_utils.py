from ollama import Client, Message  # Import Message class


def sanitized_lyrics(lyrics, ollama_url, model, message):
    client = Client(host=ollama_url)

    # Create Message objects instead of raw dictionaries
    system_msg = Message(role='system', content='You are a part of a program...')
    user_msg = Message(role='user', content=message + lyrics)

    # Call chat directly with proper message objects
    response = client.chat(
        model=model,
        messages=[system_msg, user_msg]
    )

    return response.message.content

