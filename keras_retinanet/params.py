import cv2

FONT_LIST = [cv2.FONT_HERSHEY_SIMPLEX, cv2.FONT_HERSHEY_PLAIN, cv2.FONT_HERSHEY_DUPLEX,
			cv2.FONT_HERSHEY_COMPLEX, cv2.FONT_HERSHEY_TRIPLEX, cv2.FONT_HERSHEY_COMPLEX_SMALL,
			cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, cv2.FONT_HERSHEY_SCRIPT_COMPLEX]

PUNCTUATION_LIST = [".", ",", "!", "?"]

ALPHABET = "abcdefghijklmnopqrstuvwxyz" + "".join(PUNCTUATION_LIST) + " "

INPUT_LENGTH = 100