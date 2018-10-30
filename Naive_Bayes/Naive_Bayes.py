# -*- coding: utf-8 -*-
# @Time  : 2018/10/24 20:10
# @Author : han luo
# @git   : https://github.com/Prayforhanluo

from __future__ import division


class PD(object):
    """
        A fundamental class for Probability Distribution
    """
    def __init__(self, frequency, labels):
        """
            Initial a dict, each key represent the a category and the value represent the frequency
        :param frequency:   array or list of frequency (should be int type)
        :param labels:      array or list of category (can be str , int, number ...)
        """
        try:
            assert len(frequency) == len(labels)
        except AssertionError:
            raise IOError("category number should be equal to the frequency number")

        self.pd_dict = dict(zip(labels, frequency))

    def GetProb(self, x, default=0):
        """
            Get the probability of the category x.
        :param x: category
        :param default:  if the category is not in pd dict value return to 0
        :return:  float probability
        """
        return self.pd_dict.get(x, default)

    def GetProbs(self, x):
        """
            Get the probabilities for a sequence of category
        :param x: category array or list
        :return: float probability  list
        """
        return [self.GetProb(i) for i in x]

    def SetProb(self, x, p):
        """
            Set the probability of category x
        :param x: category
        :param p: frequency
        :return:
        """
        self.pd_dict[x] = p

    def TotalFrequency(self):
        """
            sum of the Probs
        :return:
        """
        return sum(self.pd_dict.values())

    def Normalize(self, fraction=1):
        """
            make the frequency pf each category equal to 1
        :param fraction:
        :return:
        """
        total = self.TotalFrequency()
        if total == 0:
            raise ValueError('zero of total frequency ')

        factor = fraction / total
        for x in self.pd_dict:
            self.pd_dict[x] *= factor


"""
    M & M beans question.
    We have two bag of beans, 1 for 1994  and 1 for 1995.
    1994 : brown 30 yellow 20 red 20 green 10 orange 10 tan 10
    1995 : blue 24 green 20 orange 16 yellow 14 red 13 brown 13
    We take one bean from each of the two bags,one is yellow another is green.
    So, What is the probability of yellow bean comes from 1994 bag. 
"""


def M_and_M():
    """
    """
    # 1994 bag beans PD
    mix94 = PD(frequency=[30, 20, 20, 10, 10, 10], labels=['brown', 'yellow', 'red', 'green', 'orange', 'tan'])
    mix94.Normalize()
    # 1996 bag beans PD
    mix96 = PD(frequency=[24, 20, 16, 14, 13, 13], labels=['blue', 'green', 'orange', 'yellow', 'red', 'brown'])
    mix96.Normalize()
    # bag PD
    hypos = PD(frequency=[1, 1], labels=['mix94', 'mix96'])
    hypos.Normalize()

    # prior probability
    P_bag94 = hypos.GetProb('mix94')
    P_bag96 = hypos.GetProb('mix96')

    # Likelihood
    P_bag94Y_bag96G = mix94.GetProb('yellow') * mix96.GetProb('green')
    P_bag96Y_bag94G = mix94.GetProb('green') * mix96.GetProb('yellow')
    # Normalized constant
    P_yellow = P_bag94 * P_bag94Y_bag96G + P_bag96 * P_bag96Y_bag94G

    # posterior probability
    P_final = (P_bag94 * P_bag94Y_bag96G) / P_yellow

    return P_final
