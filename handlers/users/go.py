from aiogram import types
from loader import dp
from aiogram.dispatcher.filters import Command, Text
from ai_model.get_image import get_image_path
from keyboards.default import labels
from aiogram.dispatcher import FSMContext
from states.user_registration import UserRegistration
from skimage import io
import numpy as np
from ai_model import model, transformations, gradcamplusplus, get_gradcam_image
import cv2
import matplotlib.pyplot as plt

RESULT_DATA_DIR = 'results/'


@dp.message_handler(Command(commands=['go']), state=UserRegistration.t1)
async def begin_test(message: types.Message, state: FSMContext):
    path, label = get_image_path()
    img = open(path, 'br')
    await state.update_data({
        't1_image_path': path
    })
    await dp.bot.send_photo(message.chat.id, img, reply_markup=labels)
    await UserRegistration.next()


@dp.message_handler(state=UserRegistration.t11)
async def test_1(message: types.Message, state: FSMContext):
    data = await state.get_data()
    path = data['t1_image_path']
    await state.update_data({
        't1_1_answer': message.text
    })
    image = io.imread(path, as_gray=False).astype(np.float32)
    image = cv2.resize(image, (224, 224), interpolation=cv2.INTER_AREA)
    transformed_image = transformations(image).unsqueeze(0)
    output_class = model(transformed_image).argmax().item()
    gradcamed_image = get_gradcam_image(image, output_class, gradcamplusplus)
    f = plt.figure()
    save_path = RESULT_DATA_DIR + path.split('/')[2]
    plt.imsave(save_path, gradcamed_image)
    f.clear()
    plt.close(f)
    img = open(save_path, 'br')
    await dp.bot.send_photo(message.chat.id, img, reply_markup=labels)


