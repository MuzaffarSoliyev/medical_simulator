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

OUTPUTS = {
    0: 'Normal',
    1: 'Benign',
    2: 'Malignant'
}


@dp.message_handler(Command(commands=['go']), state=UserRegistration.t1)
async def begin_test(message: types.Message, state: FSMContext):
    path, label = get_image_path()
    img = open(path, 'br')
    await state.update_data({
        't1_image_path': path,
        'target_1': label
    })
    await message.answer_photo(img, reply_markup=labels)
    await UserRegistration.next()


@dp.message_handler(state=UserRegistration.t11, content_types='text')
async def test_1(message: types.Message, state: FSMContext):
    data = await state.get_data()
    path = data['t1_image_path']
    await state.update_data({
        't1_answer_1': message.text
    })
    image = io.imread(path, as_gray=False).astype(np.float32)
    image = cv2.resize(image, (224, 224), interpolation=cv2.INTER_AREA)
    transformed_image = transformations(image).unsqueeze(0)
    output_class = model(transformed_image).argmax().item()
    # gradcamed_image = get_gradcam_image(image, output_class, gradcamplusplus)
    f = plt.figure()
    save_path = RESULT_DATA_DIR + path.split('/')[2]
    plt.imsave(save_path, transformed_image)
    f.clear()
    plt.close(f)
    img = open(save_path, 'br')
    await message.answer_photo(img, reply_markup=labels, caption=f"Ответ модели: <b>{OUTPUTS[output_class]}</b>", parse_mode='HTML')
    await UserRegistration.next()


@dp.message_handler(state=UserRegistration.t12, content_types='text')
async def test_12(message: types.Message, state: FSMContext):
    answer_2 = message.text
    await state.update_data({
        't1_answer_2': answer_2
    })
    path, label = get_image_path()
    img = open(path, 'br')
    await state.update_data({
        't2_image_path': path,
        'target_2': label
    })
    await message.answer_photo(img, reply_markup=labels)
    await UserRegistration.next()


@dp.message_handler(state=UserRegistration.t2, content_types='text')
async def test_21(message: types.Message, state: FSMContext):
    data = await state.get_data()
    path = data['t2_image_path']
    await state.update_data({
        't2_answer_1': message.text
    })
    image = io.imread(path, as_gray=False).astype(np.float32)
    image = cv2.resize(image, (224, 224), interpolation=cv2.INTER_AREA)
    transformed_image = transformations(image).unsqueeze(0)
    output_class = model(transformed_image).argmax().item()
    # gradcamed_image = get_gradcam_image(image, output_class, gradcamplusplus)
    f = plt.figure()
    save_path = RESULT_DATA_DIR + path.split('/')[2]
    plt.imsave(save_path, transformed_image)
    f.clear()
    plt.close(f)
    img = open(save_path, 'br')
    await message.answer_photo(img, reply_markup=labels, caption=f"Ответ модели: <b>{OUTPUTS[output_class]}</b>", parse_mode='HTML')
    await UserRegistration.next()


@dp.message_handler(state=UserRegistration.t21, content_types='text')
async def test_22(message: types.Message, state: FSMContext):
    answer_2 = message.text
    await state.update_data({
        't2_answer_2': answer_2
    })
    path, label = get_image_path()
    img = open(path, 'br')
    await state.update_data({
        't3_image_path': path,
        'target_3': label
    })
    await message.answer_photo(img, reply_markup=labels)
    await UserRegistration.next()



@dp.message_handler(state=UserRegistration.t3, content_types='text')
async def test_31(message: types.Message, state: FSMContext):
    data = await state.get_data()
    path = data['t3_image_path']
    await state.update_data({
        't3_answer_1': message.text
    })
    image = io.imread(path, as_gray=False).astype(np.float32)
    image = cv2.resize(image, (224, 224), interpolation=cv2.INTER_AREA)
    transformed_image = transformations(image).unsqueeze(0)
    output_class = model(transformed_image).argmax().item()
    # gradcamed_image = get_gradcam_image(image, output_class, gradcamplusplus)
    f = plt.figure()
    save_path = RESULT_DATA_DIR + path.split('/')[2]
    plt.imsave(save_path, transformed_image)
    f.clear()
    plt.close(f)
    img = open(save_path, 'br')
    await message.answer_photo(img, reply_markup=labels, caption=f"Ответ модели: <b>{OUTPUTS[output_class]}</b>", parse_mode='HTML')
    await UserRegistration.next()


@dp.message_handler(state=UserRegistration.t31, content_types='text')
async def test_32(message: types.Message, state: FSMContext):
    answer_2 = message.text
    await state.update_data({
        't3_answer_2': answer_2
    })
    path, label = get_image_path()
    img = open(path, 'br')
    await state.update_data({
        't4_image_path': path,
        'target_4': label
    })
    await message.answer_photo(img, reply_markup=labels)
    await UserRegistration.next()



@dp.message_handler(state=UserRegistration.t4, content_types='text')
async def test_41(message: types.Message, state: FSMContext):
    data = await state.get_data()
    path = data['t4_image_path']
    await state.update_data({
        't4_answer_1': message.text
    })
    image = io.imread(path, as_gray=False).astype(np.float32)
    image = cv2.resize(image, (224, 224), interpolation=cv2.INTER_AREA)
    transformed_image = transformations(image).unsqueeze(0)
    output_class = model(transformed_image).argmax().item()
    # gradcamed_image = get_gradcam_image(image, output_class, gradcamplusplus)
    f = plt.figure()
    save_path = RESULT_DATA_DIR + path.split('/')[2]
    plt.imsave(save_path, transformed_image)
    f.clear()
    plt.close(f)
    img = open(save_path, 'br')
    await message.answer_photo(img, reply_markup=labels, caption=f"Ответ модели: <b>{OUTPUTS[output_class]}</b>", parse_mode='HTML')
    await UserRegistration.next()


@dp.message_handler(state=UserRegistration.t41, content_types='text')
async def test_42(message: types.Message, state: FSMContext):
    answer_2 = message.text
    await state.update_data({
        't4_answer_2': answer_2
    })
    path, label = get_image_path()
    img = open(path, 'br')
    await state.update_data({
        't5_image_path': path,
        'target_5': label
    })
    await message.answer_photo(img, reply_markup=labels)
    await UserRegistration.next()


@dp.message_handler(state=UserRegistration.t5, content_types='text')
async def test_41(message: types.Message, state: FSMContext):
    data = await state.get_data()
    path = data['t5_image_path']
    await state.update_data({
        't5_answer_1': message.text
    })
    image = io.imread(path, as_gray=False).astype(np.float32)
    image = cv2.resize(image, (224, 224), interpolation=cv2.INTER_AREA)
    transformed_image = transformations(image).unsqueeze(0)
    output_class = model(transformed_image).argmax().item()
    # gradcamed_image = get_gradcam_image(image, output_class, gradcamplusplus)
    f = plt.figure()
    save_path = RESULT_DATA_DIR + path.split('/')[2]
    plt.imsave(save_path, transformed_image)
    f.clear()
    plt.close(f)
    img = open(save_path, 'br')
    await message.answer_photo(img, reply_markup=labels, caption=f"Ответ модели: <b>{OUTPUTS[output_class]}</b>", parse_mode='HTML')
    await UserRegistration.next()


@dp.message_handler(state=UserRegistration.t51, content_types='text')
async def test_52(message: types.Message, state: FSMContext):
    answer_2 = message.text
    await state.update_data({
        't5_answer_2': answer_2
    })
    data = await state.get_data()
    await message.answer(
        f"1. target: {data['target_1']}  answer_1: {data['t1_answer_1']}  answer_2: {data['t1_answer_2']}\n" +
        f"2. target: {data['target_2']}  answer_2: {data['t2_answer_1']}  answer_2: {data['t2_answer_2']}\n" +
        f"3. target: {data['target_3']}  answer_3: {data['t3_answer_1']}  answer_2: {data['t3_answer_2']}\n" +
        f"4. target: {data['target_4']}  answer_4: {data['t4_answer_1']}  answer_2: {data['t4_answer_2']}\n" +
        f"5. target: {data['target_5']}  answer_5: {data['t5_answer_1']}  answer_2: {data['t5_answer_2']}\n\n" +
        "Чтобы ещё раз пройти тренировку наберите /go"
    )
    await state.set_state(UserRegistration.t1)



