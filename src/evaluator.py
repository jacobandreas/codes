from __future__ import print_function
from experience import Experience
from tasks.ref import RefTask
import trainer

import json
import logging
import numpy as np
import tensorflow as tf

random = np.random.RandomState(4813)

def _do_deaf_rollout(
        code_agent, desc_agent, task, rollout_ph, model, desc_to_code, session,
        config, h0, z0, fold, mode):
    demos = [task.get_demonstration(fold) for _ in range(config.trainer.n_rollout_episodes)]
    worlds = [d[0].s1 for d in demos]
    done = [False] * config.trainer.n_rollout_episodes
    episodes = [[] for i in range(config.trainer.n_rollout_episodes)]
    hs, zs = h0, z0
    dhs = h0

    empty = np.zeros(len(task.lexicon))
    empty[0] = 1
    last_desc = [empty] * config.trainer.n_rollout_episodes

    for t in range(config.trainer.n_timeout):
        hs_, zs_, qs = session.run(
                [model.tt_rollout_h, model.tt_rollout_z, model.tt_rollout_q],
                rollout_ph.feed(hs, zs, dhs, worlds, task, config))
        for i in range(config.trainer.n_rollout_episodes):
            if done[i]:
                continue

            actions = [None, None]
            actions[code_agent] = np.argmax(qs[code_agent][i, :])
            if t < len(demos[i]):
                actions[desc_agent] = demos[i][t].a[desc_agent]
            else:
                actions[desc_agent] = 0

            world_, reward, done_ = worlds[i].step(actions)

            if t < len(demos[i]):
                desc = demos[i][t].s2.l_msg[code_agent]
                last_desc[i] = desc
            else:
                desc = last_desc[i]
                #desc = np.zeros(len(task.lexicon))
                #desc[0] = 1

            #code = desc_to_code(desc, mode)[0]
            codes = desc_to_code(desc, mode)
            #code = np.mean(codes, axis=0)
            code = codes[random.randint(len(codes))]
            #code = np.random.choice(desc_to_code(desc, mode))

            #print(str(desc))
            #print(str(code[:5]))
            zs_[desc_agent][i, :] = code

            episodes[i].append(Experience(
                worlds[i], None, tuple(actions), world_, None, reward, done_))
            worlds[i] = world_
            done[i] = done_

        hs = hs_
        zs = zs_
        if all(done):
            break

    return (sum(e.r for ep in episodes for e in ep) * 1. / 
                config.trainer.n_rollout_episodes, 
            sum(ep[-1].s2.success for ep in episodes) * 1. /
                config.trainer.n_rollout_episodes)

def _do_tr_rollout(
        code_agent, desc_agent, task, rollout_ph, model, desc_model,
        desc_to_code, code_to_desc, session, config, h0, z0, fold, mode):
    worlds = [task.get_instance(fold) for _ in range(config.trainer.n_rollout_episodes)]
    done = [False] * config.trainer.n_rollout_episodes
    episodes = [[] for i in range(config.trainer.n_rollout_episodes)]
    hs, zs = h0, z0
    dhs = h0
    for t in range(config.trainer.n_timeout):
        hs_, zs_, qs = session.run(
                [model.tt_rollout_h, model.tt_rollout_z, model.tt_rollout_q],
                rollout_ph.feed(hs, zs, dhs, worlds, task, config))
        dhs_, dqs = session.run(
                [desc_model.tt_rollout_h, desc_model.tt_rollout_q],
                rollout_ph.feed(hs, zs, dhs, worlds, task, config))
        for i in range(config.trainer.n_rollout_episodes):
            if done[i]:
                continue

            actions = [None, None]
            actions[code_agent] = np.argmax(qs[code_agent][i, :])
            actions[desc_agent] = np.argmax(dqs[desc_agent][i, :])

            world_, reward, done_ = worlds[i].step(actions)

            code = desc_to_code(world_.l_msg[code_agent], mode)[0]
            zs_[desc_agent][i, :] = code

            l_words = code_to_desc(zs_[code_agent][i, :], mode)[:5]
            l_msg = np.zeros(len(task.lexicon))
            for l_word in l_words:
                l_msg[task.lexicon.index(l_word)] += 1
            l_msg /= np.sum(l_msg)

            world_.l_msg = list(world_.l_msg)
            world_.l_msg[desc_agent] = l_msg
            world_.l_msg = tuple(world_.l_msg)

            episodes[i].append(Experience(
                worlds[i], None, tuple(actions), world_, None, reward, done_))
            worlds[i] = world_
            done[i] = done_

            if config.evaluator.simulate_l:
                assert False

        hs = hs_
        zs = zs_
        dhs = dhs_
        if all(done):
            break

    return (sum(e.r for ep in episodes for e in ep) * 1. / 
                config.trainer.n_rollout_episodes, 
            sum(ep[-1].s2.success for ep in episodes) * 1. /
                config.trainer.n_rollout_episodes)

def run(task, rollout_ph, replay_ph, reconst_ph, model, desc_model,
        lexicographer, session, config, fold="test"):
    h0, z0, _ = session.run(model.zero_state(1, tf.float32))

    if isinstance(task, RefTask):
        count = config.evaluator.n_episodes
    else:
        #count = 100
        count = 500

    with open(config.experiment_dir + "/eval.txt", "w") as eval_f:
        task.reset_test()
        l_l_score = np.asarray([0., 0.])
        for i in range(count):
            score = trainer._do_rollout(
                    task, rollout_ph, model, desc_model, [], [], session,
                    config, 10000, h0, z0, fold, use_desc=True)
            l_l_score += score
        l_l_score /= count
        logging.info("[l,l]  \t%s" % str(l_l_score))
        print("l only:", file=eval_f)
        print(l_l_score, file=eval_f)
        task.reset_test()
        c_c_score = np.asarray([0., 0.])
        for i in range(count):
            score = trainer._do_rollout(
                    task, rollout_ph, model, desc_model, [], [], session,
                    config, 10000, h0, z0, fold, use_desc=False)
            c_c_score += score
        c_c_score /= count
        logging.info("[c,c]  \t%s" % str(c_c_score))
        logging.info("")
        print("c only:", file=eval_f)
        print(c_c_score, file=eval_f)
        for mode in ["fkl", "rkl", "pmi", "dot", "rand"]:
            print(mode + ":", file=eval_f)
            task.reset_test()
            c_l_score = np.asarray([0., 0.])
            for i in range(count):
                if isinstance(task, RefTask):
                    score = _do_tr_rollout(
                            0, 1, task, rollout_ph, model, desc_model, lexicographer.l_to_c,
                            lexicographer.c_to_l, session, config, h0, z0, fold, mode)
                else:
                    score = _do_deaf_rollout(
                            0, 1, task, rollout_ph, model, lexicographer.l_to_c, 
                            session, config, h0, z0, fold, mode)
                c_l_score += score
            c_l_score /= count
            print("(c, l)", c_l_score, file=eval_f)
            logging.info("[c,l:%s]  \t%s" % (mode, str(c_l_score)))

            if isinstance(task, RefTask):
                task.reset_test()
                l_c_score = np.asarray([0., 0.])
                for i in range(count):
                    score = _do_tr_rollout(
                            1, 0, task, rollout_ph, model, desc_model, lexicographer.l_to_c,
                            lexicographer.c_to_l, session, config, h0, z0, fold, mode)
                    l_c_score += score
                l_c_score /= count
                print("(l, c)", l_c_score, file=eval_f)
                logging.info("[l,c:%s]  \t%s" % (mode, str(l_c_score)))

            logging.info("")
