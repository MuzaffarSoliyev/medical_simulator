from aiogram.types import (ReplyKeyboardMarkup, KeyboardButton)


modes = ReplyKeyboardMarkup(
    keyboard=[
        [
            KeyboardButton(text='Рак головного мозга')
        ],
        [
            KeyboardButton(text='Рак груди')
        ],
    ],
    resize_keyboard=True
)