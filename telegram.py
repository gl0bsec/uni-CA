import pandas as pd
import asyncio
import csv
import sqlite3
from telethon import TelegramClient
from telethon.errors import FloodWaitError
from telethon.tl.functions.contacts import SearchRequest
from telethon.tl.types import InputPeerEmpty, Message
from time import sleep
import nest_asyncio
import os
from datetime import datetime

# Apply nest_asyncio to allow nested use of asyncio.run()
nest_asyncio.apply()

# Create the client with a timeout to avoid the locked database issue
client = TelegramClient(session_name, api_id, api_hash, connection_retries=2)

async def search_channels(client, keyword, min_participants_count, search_limit, top_n):
    await client.start(phone_number)
    # Perform the search
    result = await client(SearchRequest(
        q=keyword,  # The keyword to search for
        limit=search_limit  # Number of results to return
    ))

    # Extract channel information
    channels = []
    for chat in result.chats:
        participants_count = chat.participants_count if hasattr(chat, 'participants_count') else 0
        if (hasattr(chat, 'title') and keyword.lower() in chat.title.lower() or
            hasattr(chat, 'about') and keyword.lower() in chat.about.lower()) and \
           participants_count >= min_participants_count:
            channels.append({
                'id': chat.id,
                'title': chat.title,
                'description': chat.about if hasattr(chat, 'about') else '',
                'participants_count': participants_count,
                'username': chat.username if hasattr(chat, 'username') else ''
            })

    # Sort channels by participants count and return top N
    channels.sort(key=lambda x: x['participants_count'], reverse=True)
    return channels[:top_n]

async def extract_telegram_channel_to_csv(api_id, api_hash, phone_number, channel_username, output_file, start_date=None, end_date=None):
    # Convert start_date and end_date to datetime objects if provided
    if start_date:
        start_date = datetime.strptime(start_date, '%Y-%m-%d')
    if end_date:
        end_date = datetime.strptime(end_date, '%Y-%m-%d')

    client = TelegramClient('session_name2', api_id, api_hash)
    await client.start(phone_number)

    with open(output_file, mode='w', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['id', 'date', 'message', 'sender_id', 'sender_name', 'reply_to_msg_id', 'media', 'views', 'forwards', 'edit_date']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        async for message in client.iter_messages(channel_username):
            # Skip messages outside the specified date range
            if start_date and message.date < start_date:
                continue
            if end_date and message.date > end_date:
                continue

            writer.writerow({
                'id': message.id,
                'date': message.date,
                'message': message.message,
                'sender_id': message.sender_id,
                'sender_name': (await client.get_entity(message.sender_id)).username if message.sender_id else None,
                'reply_to_msg_id': message.reply_to_msg_id,
                'media': message.media if message.media else None,
                'views': message.views if hasattr(message, 'views') else None,
                'forwards': message.forwards if hasattr(message, 'forwards') else None,
                'edit_date': message.edit_date
            })

    print(f"Messages have been successfully saved to {output_file}")

    # Disconnect the client
    await client.disconnect()

# Example usage:
# asyncio.run(extract_telegram_channel_to_csv(api_id, api_hash, phone_number, 'MeghUpdates', 'output_file.csv', None, None))

# %%
