# -*- coding: utf-8 -*-
"""Chatbot.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/10ogAJNgHbWuInfWBqgdvlVS2I4VGZKI0
"""

import re
import random

def message_probability(user_message, recognised_words, single_response=False, required_words=[]):
    recognized_word_count = sum(1 for word in user_message if word in recognised_words)
    if len(recognised_words) == 0:
        return 0
    percentage = recognized_word_count / len(recognised_words)
    has_required_words = all(word in user_message for word in required_words)
    if has_required_words or single_response:
        return int(percentage * 100)
    else:
        return 0

def check_all_messages(message):
    highest_prob_list = {}
    def response(bot_response, list_of_words, single_response=False, required_words=[]):
        nonlocal highest_prob_list
        if message:
            highest_prob_list[bot_response] = message_probability(message, list_of_words, single_response, required_words)

    response('Hello! I\'m your professional chatbot.', ['hello', 'hi', 'hey', 'heyo'], single_response=True)
    response('I am programmed to assist you with various tasks. How old are you?', ['want', 'can', 'do', 'you'], required_words=['age','want'])
    response('As a chatbot, I don\'t have a gender. What else would you like to know?', ['what', 'your', 'name'], required_words=['name'])
    response('My name is Chatbot. How can I help you?', ['name', 'you'], required_words=['name'])
    response('I am here to assist you 24/7. How can I help you today?', ['help', 'information'], required_words=['information'])
    response('I\'m sorry, I don\'t understand that. Could you please rephrase?', ['support', 'troubleshooting'], required_words=['troubleshooting'])
    response('Okay, have a great day!', [], single_response=True)

    return max(highest_prob_list, key=highest_prob_list.get)

def get_response(user_input):
    split_message = re.split(r'\s+|[,;?!.-]\s*', user_input.lower())
    response = check_all_messages(split_message)
    return response

def exit_prompt():
    while True:
        exit_input = input("Chatbot: Is there anything else I can assist you with? (Yes/No): ").lower()
        if exit_input in ['yes', 'y']:
            return False
        elif exit_input in ['no', 'n']:
            print("Chatbot: Okay, have a great day!")
            return True
        else:
            print("Chatbot: I'm sorry, I didn't understand that. Please enter 'Yes' or 'No'.")

print("Bot: Hi there! What can you do?")

while True:
    user_input = input('You: ')
    print('Chatbot:', get_response(user_input))
    if exit_prompt():
        break