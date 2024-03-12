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

curr_number = [np.random.randint(1, 100)]

lang_clf = AudioLangClassifier()
transcriptor = Transcriptor()

config = yaml.safe_load(open(os.path.join('.', 'config.yaml'), 'r'))
TG_API = config.get('API_TOKEN')

app = FastAPI(
    title='Studying french telegram bot'
)


@app.post('/')
async def read_root(request: Request):


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


async def get_audio_url(file_id):
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


async def validate(curr_lang, chat_id, waveform, sample_rate):
    if curr_lang != 'French':
        bot_speech = f'Sorry, but you spoke kind a {curr_lang}. Please, speak French.}'
        await bot_says(bot_speech, chat_id)

    elif curr_lang == 'French':
        curr_transcription = transcriptor.get_transcription(waveform, sample_rate)

        if curr_transcription[0].find(f'{curr_number[0]}') == -1:
            bot_speech = f'Try again! Your answer is: {curr_transcription[0]}'
            await bot_says(bot_speech, chat_id)

        else:
            curr_number[0] = np.random.randint(1, 100)
            bot_speech = f'Nice!'
            await bot_says(bot_speech, chat_id)


async def bot_says(text: str, chat_id: int):
    async with aiohttp.ClientSession() as session:
        url = f'https://api.telegram.org/bot{TG_API}/sendMessage'
        async with session.post(url=url, data={
            'chat_id': chat_id,
            'text': f'{text}',
        }) as response:
            res = await response.json()


if __name__ == '__main__':
    uvicorn.run('main:app', host='0.0.0.0', port=8000, reload=False)
