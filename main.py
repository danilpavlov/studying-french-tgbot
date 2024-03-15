import os
import uuid

import aiofiles
import aiohttp
import uvicorn
import yaml
from fastapi import FastAPI, Request

from schemas import Answer
from models import Transcriptor, AudioLangClassifier, process_audio

import numpy as np
import torch


LIMIT_NUMBER = 20
curr_number = [np.random.randint(1, LIMIT_NUMBER)]

lang_clf = AudioLangClassifier()
transcriptor = Transcriptor()

config = yaml.safe_load(open(os.path.join('.', 'config.yaml'), 'r'))
TG_API = config.get('API_TOKEN')

app = FastAPI(
    title='Studying french telegram bot'
)


@app.post('/')
async def read_root(request: Request):
    """
    Parse the incoming JSON data from the request.
    Extract the file ID from the message's voice or audio attribute.
    If a valid file ID is found, retrieve the audio file's URL asynchronously.
    Extract the chat ID from the message.
    Process the audio waveform and obtain its sample rate.
    Predict the language of the spoken audio.
    Validate the spoken audio based on the predicted language.

    :param
        * request (Request): The incoming HTTP request object.
    """
    json = await request.json()
    print(json)
    obj = Answer.model_validate(json)
    if obj.message.voice:
        file_id = obj.message.voice.file_id
    elif obj.message.audio:
        file_id = obj.message.audio.file_id
    else:
        file_id = -1

    if file_id != -1:
        audio_url = await get_audio_url(file_id)

        chat_id = (
            obj.message.chat.id
            if obj.message.chat.id is not None
            else obj.message.from_f.chat_id
        )

        waveform, sample_rate = process_audio(audio_url)
        os.remove(audio_url)

        curr_lang = lang_clf.predict(waveform, sample_rate)

        await validate(curr_lang, chat_id, waveform, sample_rate)


async def get_audio_url(file_id: int) -> str:
    """
    Construct the URL to request file information from the Telegram Bot API.
    Perform an asynchronous HTTP request to obtain file information.
    If the response is successful, extract the file path and download the audio file.
    Save the downloaded audio file with a unique filename and return its URL.

    :param
        * file_id (int): The unique identifier of the audio file to retrieve from the Telegram Bot API.

    :return
        * audio_url (str): The URL of the downloaded audio file.
    """
    url = f'https://api.telegram.org/bot{TG_API}/getFile'
    data = {'file_id': file_id}
    filename = uuid.uuid4().hex

    async with aiohttp.ClientSession() as session:
        async with session.post(url, data=data) as response:
            res_file_info = await response.json()

            print(res_file_info)
            if res_file_info.get('ok'):
                path = res_file_info['result']['file_path']
                ext = path.split('.')[-1]

                audio_url = f'https://api.telegram.org/file/bot{TG_API}/{path}'

                async with session.get(audio_url) as download_file:
                    audio_url = f'{filename}.{ext}'
                    async with aiofiles.open(audio_url, mode='wb') as f:
                        contents = await download_file.read()
                        await f.write(contents)
    return audio_url


async def validate(curr_lang: str, chat_id: int, waveform: torch.Tensor, sample_rate: int) -> None:
    """
    Check if the current language is not French. If not, prompt the user to speak French.
    If the current language is French, transcribe the provided waveform.
    If the transcription does not contain the expected number, prompt the user to try again.
    If the transcription contains the expected number, generate a new random number and acknowledge the user.

    :param
        * curr_lang (str): The current language spoken by the user.
        * chat_id (int): The unique identifier of the chat session.
        * waveform (torch.Tensor): The audio waveform of the user's speech.
        * sample_rate (int): The sample rate of the audio waveform.
    """
    if curr_lang != 'French':
        bot_speech = f'''
        Sorry, but you spoke kind a {curr_lang}. Please, speak French.Current number: {curr_number}
        '''
        await bot_says(bot_speech, chat_id)

    elif curr_lang == 'French':
        curr_transcription = transcriptor.get_transcription(waveform, sample_rate)

        if curr_transcription.find(f'{curr_number[0]}') == -1:
            bot_speech = f'''
            Try again! Your answer is: {curr_transcription} Current number: {curr_number}
            '''
            await bot_says(bot_speech, chat_id)

        else:
            curr_number[0] = np.random.randint(1, LIMIT_NUMBER)
            bot_speech = f'''
            Nice! Current number: {curr_number}
            '''
            await bot_says(bot_speech, chat_id)


async def bot_says(text: str, chat_id: int) -> None:
    """
    Establish an asynchronous HTTP client session.
    Construct the URL to send a message using the Telegram Bot API.
    Perform an asynchronous POST request to send the message to the specified chat.
    Handle the response from the Telegram API.

    :param
        * text (str): The text message to be sent by the bot.
        * chat_id (int): The unique identifier of the chat session.
    """
    async with aiohttp.ClientSession() as session:
        url = f'https://api.telegram.org/bot{TG_API}/sendMessage'
        async with session.post(url=url, data={
            'chat_id': chat_id,
            'text': f'{text}',
        }) as response:
            res = await response.json()


if __name__ == '__main__':
    uvicorn.run('main:app', host='0.0.0.0', port=8000, reload=False)
