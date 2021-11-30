from aiogram.types import (ReplyKeyboardMarkup, KeyboardButton)


labels = ReplyKeyboardMarkup(
    keyboard=[
        [
            KeyboardButton(text='Benign')
        ],
[
            KeyboardButton(text='Malignant')
        ],
[
            KeyboardButton(text='Normal')
        ]
    ],
    resize_keyboard=True
)