class TrainConfig:
    NAME = "custom_voice"
    GENDER = "F"
    TARGET_WAV_PATH = "voices/source.wav"
    REFERENCE_STYLE = "auto"
    SEED = 42
    SPEED = 1.05
    NUM_STEPS = 3000
    LEARNING_RATE = 0.0002
    VOCODER_STEPS = 5
    SAVE_STEPS = 500
    EARLY_STOP_LOSS_THRESHOLD = 0.24
