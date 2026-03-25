# config.py
CHARSET = r"0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ!\"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"
TARGET_VEHICLES = {2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}

class OCRResult:
    def __init__(self, text, confidence, update_count):
        self.text = text
        self.confidence = confidence
        self.update_count = update_count
        self.is_logged = False

class Detection:
    def __init__(self, box, score, class_id):
        self.box = box
        self.score = score
        self.class_id = class_id