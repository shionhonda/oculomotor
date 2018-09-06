# -*- coding: utf-8 -*-
"""
 Evaluation script for final evaluation.
"""

import argparse
import os

from agent import Agent
from functions import BG, FEF, LIP, PFC, Retina, SC, VC, HP, CB
from oculoenv import Environment
from oculoenv import PointToTargetContent, ChangeDetectionContent, OddOneOutContent, VisualSearchContent, \
    MultipleObjectTrackingContent, RandomDotMotionDiscriminationContent
from logger import Logger

"""
POINT_TO_TARGET                 : id=1, difficulty_range=3 (0, 1, 2)
CHANGE_DETECTION                : id=2, difficulty_range=5 (0, 2, 4)
ODD_ONE_OUT                     : id=3, difficulty_range=0 
VISUAL_SEARCH                   : id=4, difficulty_range=6 (0, 2, 5)
MULTIPLE_OBJECT_TRACKING        : id=5, difficulty_range=6 (0, 2, 5)
RANDOM_DOT_MOTION_DISCRIMINATION: id=6  difficulty_range=5 (0, 2, 4)
"""

TASK1_DURATION = 60*30
TASK2_DURATION = 60*30
TASK3_DURATION = 60*30
TASK4_DURATION = 60*30
TASK5_DURATION = 60*40
TASK6_DURATION = 60*30


content_class_names = [
    "PointToTargetContent",
    "ChangeDetectionContent",
    "OddOneOutContent",
    "VisualSearchContent",
    "MultipleObjectTrackingContent",
    "RandomDotMotionDiscriminationContent"
]

class TrialResult(object):
    def __init__(self, content_id, difficulty, info):
        self.content_id = content_id
        self.difficulty = difficulty
        if info['result'] == 'success':
            self.succeed = 1
        else:
            self.succeed = 0
        self.reaction_step = info['reaction_step']

    def __lt__(self, other):
        if self.content_id != other.content_id:
            return self.content_id < other.content_id
        else:
            return self.difficulty < other.difficulty

    def get_string(self):
        return "{}, {}, {}, {}".format(self.content_id,
                                       self.difficulty,
                                       self.succeed,
                                       self.reaction_step)


class EvaluationTask(object):
    def __init__(self, content_id, difficulty, duration):
        self.content_id = content_id
        self.difficulty = difficulty
        self.duration = duration

    def evaluate(self, agent):
        print("content:{} difficulty:{} start".format(self.content_id, self.difficulty))
        
        content_class_name = content_class_names[self.content_id-1]
        content_class = globals()[content_class_name]
        if self.difficulty >= 0:
            content = content_class(difficulty=self.difficulty)
        else:
            content = content_class()

        env = Environment(content)
        obs = env.reset()

        reward = 0
        done = False

        task_reward = 0

        results = []

        for i in range(self.duration):
            image, angle = obs['screen'], obs['angle']
            # Choose action by the agent's decision
            action = agent(image, angle, reward, done)
            # Foward environment one step
            obs, reward, done, info = env.step(action)

            if 'result' in info:
                result = TrialResult(self.content_id,
                                     self.difficulty,
                                     info)
                results.append(result)
            
            task_reward += reward

            assert(done is not True)

        print("content:{} difficulty:{} end, reward={}".format(self.content_id,
                                                               self.difficulty,
                                                               task_reward))
        return results, task_reward


tasks = [
    EvaluationTask(content_id=1, difficulty=0, duration=TASK1_DURATION),
    EvaluationTask(content_id=2, difficulty=0, duration=TASK2_DURATION),
    EvaluationTask(content_id=4, difficulty=0, duration=TASK4_DURATION),
    EvaluationTask(content_id=5, difficulty=0, duration=TASK5_DURATION),
    EvaluationTask(content_id=6, difficulty=0, duration=TASK6_DURATION),
    
    EvaluationTask(content_id=1, difficulty=1, duration=TASK1_DURATION),
    EvaluationTask(content_id=2, difficulty=2, duration=TASK2_DURATION),
    EvaluationTask(content_id=4, difficulty=2, duration=TASK4_DURATION),
    EvaluationTask(content_id=5, difficulty=2, duration=TASK5_DURATION),
    EvaluationTask(content_id=6, difficulty=2, duration=TASK6_DURATION),
    
    EvaluationTask(content_id=1, difficulty=2, duration=TASK1_DURATION),
    EvaluationTask(content_id=2, difficulty=4, duration=TASK2_DURATION),
    EvaluationTask(content_id=4, difficulty=5, duration=TASK4_DURATION),
    EvaluationTask(content_id=5, difficulty=5, duration=TASK5_DURATION),
    EvaluationTask(content_id=6, difficulty=4, duration=TASK6_DURATION),

    EvaluationTask(content_id=3, difficulty=-1, duration=TASK3_DURATION)
]


def save_results(all_results, log_path):
    """ Save result into csv file. """
    
    if not os.path.exists(log_path):
      os.makedirs(log_path)
      
    file_path = "{}/evaluation.csv".format(log_path)
      
    f = open(file_path, mode='w')
    sorted_results = sorted(all_results)

    with open(file_path, mode='w') as f:
        for result in sorted_results:
            f.write(result.get_string())
            f.write("\n")
            

def evaluate(logger, log_path):
    retina = Retina()
    lip = LIP()
    vc = VC()
    pfc = PFC()
    fef = FEF()
    bg = BG()
    sc = SC()
    hp = HP()
    cb = CB()
    
    agent = Agent(
        retina=retina,
        lip=lip,
        vc=vc,
        pfc=pfc,
        fef=fef,
        bg=bg,
        sc=sc,
        hp=hp,
        cb=cb
    )

    #bg.load_model("model.pkl")

    total_reward = 0

    all_results = []
    
    for i, task in enumerate(tasks):
        results, task_reward = task.evaluate(agent)
        all_results += results
        total_reward += task_reward
        logger.log("evaluation_reward", total_reward, i)

    # Save result csv
    save_results(all_results, log_path)
    
    print("evaluation finished:")
    logger.close()
    
    
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log_file", help="Log file name", type=str, default="evaluate0")
    
    args = parser.parse_args()
    
    log_file = args.log_file

    # Log is stored 'log' directory
    log_path = "log/{}".format(log_file)

    logger = Logger(log_path)

    # Start evaluation
    evaluate(logger, log_path)


if __name__ == '__main__':
    main()
