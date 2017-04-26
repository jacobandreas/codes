from __future__ import print_function
from tasks.ref import RefTask
import trainer

import logging
import numpy as np
import tensorflow as tf

random = np.random.RandomState(5892)

def run(task, rollout_ph, model, desc_model, lexicographer, session, config):
    h0, z0, _ = session.run(model.zero_state(1, tf.float32))
    demonstrations = [
            task.get_demonstration("val") 
            for _ in range(config.trainer.n_batch_episodes)]

    #randomized = []
    #for _ in range(config.trainer.n_batch_episodes):
    #    ep = task.get_demonstration("val")
    #    for transition in ep:
    #        for i_agent in range(task.n_agents):
    #            transition.s1.l_msg = list(transition.s1.l_msg)
    #            transition.s1.l_msg[i_agent] = np.zeros(len(task.lexicon))
    #            transition.s1.l_msg[i_agent][
    #                    random.randint(len(task.lexicon))] = 1
    #            transition.s1.l_msg = tuple(transition.s1.l_msg)
    #    randomized.append(ep)

    #rollouts = []
    #for _ in range(config.trainer.n_batch_episodes):
    #    trainer._do_rollout(
    #            task, rollout_ph, model, desc_model, rollouts, [], session,
    #            config, 10000, h0, z0, fold="val")

    speaker_agree = {"human": 0, "random": 0}
    actor_agree = {"human": 0, "random": 0}
    count = {"human": 0, "random": 0}

    for seqs, source in [(demonstrations, "human")]: #, (randomized, "random")]:
        for ep in seqs:
            hs = h0
            for t, transition in enumerate(ep):
                hs_, scores = session.run(
                    [desc_model.tt_rollout_h, desc_model.tt_rollout_q],
                    rollout_ph.feed(h0, z0, hs, [transition.s1], task, config))
                probs = [np.exp(sc[0, :]) / np.exp(sc[0, :]).sum() for sc in scores]
                if isinstance(task, RefTask):
                    if t > 0:
                        count[source] += 1
                        if np.argmax(probs[1]) == transition.a[1]:
                            actor_agree[source] += 1
                        #actor_agree[source] += probs[1][transition.a[1]]
                else:
                    pass
                    #assert False
                hs = hs_

    logging.info("[cal]  \t" + str(actor_agree))
    logging.info("")

    with open(config.experiment_dir + "/calibrate.txt", "w") as cal_f:
        print("speaker agreement {}".format(speaker_agree), file=cal_f)
        print("actor agreement {}".format(actor_agree), file=cal_f)
        print("count {}".format(count), file=cal_f)
