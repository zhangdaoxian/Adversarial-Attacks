"""object that store things need to be record by adversarial"""
from utils.math_func import compute_psnr
import numpy as np


class LatentAdversarial:
    def __init__(self, clean_img, is_target_attack=True, target_codeword = None, latent_model=None, clean_img_path=None,
                 better_criterion='psnr'):
        self.__clean_img = clean_img
        self.__model = latent_model
        self.__is_target_attack = is_target_attack
        self.__target_codeword = target_codeword
        self.__clean_img_path = clean_img_path
        assert better_criterion in ['psnr', 'linf']
        self.__better_criterion = better_criterion
        self.__best_distance = np.inf  # may need to change this accodring to better_criterion that comes latter
        self.__best_adv = None # may need to change according to attacks that come later
        self.__clean_codeword = latent_model.predict(clean_img, output_type='codewords')
        if target_codeword is not None and hasattr(latent_model, '_codeword2label'):
            self.__target_class = latent_model._codeword2label(latent_model._decoding(target_codeword)[:4])
            self.__target_error = np.mod(self.__clean_codeword - target_codeword, 2)
            # target_codeword could be illegal, while latent_model doesn't output undecoded codeword yet, so decode target_codeword here
            self.__target_codeword = latent_model._decoding(target_codeword)
    def new_perturbed(self, perturbed):
        # when a new perturbed was generated, use this to see if it is better and whether records needs to be updated
        if not self.__is_target_attack:
            raise TypeError('Non-target attack not implement yet')
        predict_codeword = self.__model.predict(perturbed, output_type='codewords')
        # hamming_dis = np.sum(np.mod(predict_codeword + ori_codeword, 2))
        is_adv = (self.__target_codeword == predict_codeword).all()
        linf = np.max(np.abs(self.__clean_img - perturbed))
        psnr = -compute_psnr(self.__clean_img, perturbed) # set it to minus so aligned with other distance that smaller is better
        if self.__better_criterion == 'psnr':
            new_distance = psnr
        elif self.__better_criterion == 'linf':
            new_distance = linf
        else:
            new_distance = np.inf
        # pick smallest distance
        if is_adv and self.__best_distance > new_distance:
            self.__best_distance = new_distance
            self.__best_adv = perturbed
        # todo think about other things need record, like eps or something
        return is_adv, linf, psnr

    @property
    def best_adv(self):
        return self.__best_adv

    @property
    def better_adv_criterion(self):
        return self.__better_criterion

    @property
    def clean_img(self):
        return self.__clean_img

    @property
    def clean_codeword(self):
        return self.__clean_codeword

    @property
    def target_error(self):
        return self.__target_error

    @property
    def best_distance(self):
        return self.__best_distance