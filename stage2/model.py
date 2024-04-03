import sys
sys.path.append(r"/app/deepvecfont")
from models.imgsr.modules import TrainOptions, create_model


def create_sr_model():
    imgsr_opt = TrainOptions().parse()
    imgsr_opt.isTrain = False
    imgsr_opt.batch_size = 1
    imgsr_opt.phase = 'test'
    imgsr_model = create_model(imgsr_opt)
    imgsr_model.setup(imgsr_opt)

    return imgsr_model
