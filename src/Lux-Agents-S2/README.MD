# Training Metrics with Stable Baselines 3

Training metrics can vary significantly based on the chosen algorithm, but certain metrics remain universally applicable, regardless of the algorithm used. When developing a Reinforcement Learning (RL) agent responsible for maneuvering a heavy robot to gather ice and sustain its factory, the following metrics could be considered:

| Metric      | Description |
| ----------- | ----------- |
| **ice_dug** | Total amount of ice dug by the agent across all units. |
| **water_produced** | Total amount of water produced by the factory. |
| **action_queue_updates_success** | Number of successful updates to the agent's action queue. |
| **action_queue_updates_total** | Total number of attempted updates to the agent's action queue. The less the better. |
| **ep_len_mean** | Average episode length over a batch of episodes. |
| **ep_rew_mean** | Average episode reward over a batch of episodes. |
| **fps** | Frames per second, indicating how many iterations (game steps) were processed per second during training. |
| **iterations** | Number of training iterations completed. |
| **time_elapsed** | Time elapsed in seconds since the start of training. |
| **total_timesteps** | Total number of game steps (timesteps) processed during training. |

Algorithm specific (PPO in this case) metrics are the following:

| Metric      | Description |
| ----------- | ----------- |
| **approx_kl** | Approximate Kullback-Leibler (KL) divergence between the new and old policy distributions. |
| **clip_fraction** | Fraction of policy updates that were clipped due to the clipping constraint. |
| **clip_range** | Clipping range used for PPO updates. |
| **entropy_loss** | The entropy of the policy distribution. Higher entropy encourages exploration. |
| **explained_variance** | Proportion of the variance in the returns explained by the value function. |
| **learning_rate** | Learning rate used for optimization. |
| **loss** | Combined loss of the policy and value networks. |
| **n_updates** | Number of policy updates performed during the current iteration. |
| **policy_gradient_loss** | Loss due to the policy gradient component of the PPO update. |
| **value_loss** | Loss due to the value function component of the PPO update. |

A single iteration therefore looks as the following:

| Metric                           | Value         |
|----------------------------------|---------------|
| rollout/ep_len_mean              | 200           |
| rollout/ep_rew_mean              | 0.056         |
| time/fps                         | 876           |
| time/iterations                  | 8             |
| time/time_elapsed                | 36            |
| time/total_timesteps             | 32000         |
| train/approx_kl                  | 0.0005054677  |
| train/clip_fraction              | 0             |
| train/clip_range                 | 0.2           |
| train/entropy_loss               | -2.48         |
| train/explained_variance         | 0.924         |
| train/learning_rate              | 0.0003        |
| train/loss                       | -0.00543      |
| train/n_updates                  | 14            |
| train/policy_gradient_loss       | -0.00253      |
| train/value_loss                 | 1.45e-05      |
| train_metrics/action_queue_updates_success | 138 |
| train_metrics/action_queue_updates_total   | 175 |
| train_metrics/ice_dug            | 2             |
| train_metrics/water_produced     | 0             |

The final results are then compressed using `numpy .npz` file containing the **ice_dug** at specific timesteps and the **episode_lengths**.

A `.npz` contains the following:

```python
['timesteps', 'results', 'ep_lengths']

timesteps = 
[  96000  192000  288000  384000  480000  576000  672000  768000  864000
  960000 1056000 1152000 1248000 1344000 1440000 1536000 1632000 1728000
 1824000 1920000 2016000 2112000 2208000 2304000 2400000 2496000 2592000
 2688000 2784000 2880000 2976000 3072000 3168000 3264000 3360000 3456000
 3552000 3648000 3744000 3840000 3936000 4032000 4128000 4224000 4320000
 4416000 4512000 4608000 4704000 4800000 4896000 4992000 5088000 5184000
 5280000 5376000 5472000 5568000 5664000 5760000 5856000 5952000 6048000
 6144000]

results = 
[[   0    0    0    0    0]
 [   0    0    0    0    0]
 [   0    0    0    0    0]
 [   0    0    0    0    0]
 [   0    0    0   10    0]
 [   0    0    0    0    0]
 [   0    1    0   40   50]
 [   0  146  203    0  637]
 [   0    0    1    0    0]
 [   0    0    0   15    0]
 [   0    0    0  441    0]
 [   0    2   15   94    0]
 [   0    0  119  194  921]
 [   0    0    0  322   57]
 [   0  151  223  313  271]
 [   0    0   26  313    0]
 [   0   26   43    0  489]
 [   0  138  594  603  203]
 [   0    0   10   47    0]
 [   0    0    0  276    1]
 [   0    0  247    0  892]
 [   0  223    0  656 1805]
 [   0    0  142 1593 1312]
 [ 132    0  924 1415 1112]
 [   0    0    0  161    0]
 [  11  671 1862 1925 1607]
 [   0    0    0 1605 1628]
 [   0  228    0 1010 1141]
 [   0    0    0 1310  416]
 [   0   10 2147 1809 1624]
 [   0  176 1189 1656  945]
 [   0   78  213 1493    0]
 [   0    0   83    0 1290]
 [   0    0  213  307 1492]
 [   0    0    0    0 1327]
 [   0  145  499  657  709]
 [   0    0    0    0 2044]
 [   0 1050 2236  894 1164]
 [   0    4 1862 1084 1234]
 [ 104  660 1263 1375  772]
 [   8    0    5 1842    0]
 [   0    0  338 1238 2231]
 [   0    0  429 2329    0]
 [   0  422 1480 1170 2033]
 [   0    0 2051 1903 1853]
 [ 218  315 1812  959  145]
 [   0  420 1905 1748 1655]
 [   8   34  353 1750 1222]
 [   0   43 1464 1280 1263]
 [1571 1495 1737 1822 2128]
 [   0  768 1733  866  431]
 [2436 2159 1753  876 2283]
 [ 421 2278 1867 2227 2069]
 [   0  681 1905 2102 1837]
 [   0 2555 2190 2362  156]
 [   0  622 1575 2187 1677]
 [2171 1974 1795 1622 1554]
 [ 327 2249 1698 2342    2]
 [   0    0 2231 1819 2207]
 [ 406 1772 2403 2514 1658]
 [  11 2322 1877 1603 2044]
 [ 239 2631  890 2446    4]
 [2264 2169 1084 2342 2304]
 [   0 1991 2225  754 1977]]

 ep_lengths = 
[[ 301  301  301  301  301]
 [ 301  301  301  301  301]
 [ 301  301  301  301  301]
 [ 301  301  301  301  301]
 [ 301  301  301  306  301]
 [ 301  301  301  301  301]
 [ 301  301  301  336  346]
 [ 301  441  496  301  906]
 [ 301  301  301  301  301]
 [ 301  301  301  316  301]
 [ 301  301  301  716  301]
 [ 301  301  316  386  301]
 [ 301  301  416  486 1000]
 [ 301  301  301  611  356]
 [ 301  446  516  601  561]
 [ 301  301  326  601  301]
 [ 301  326  336  301  771]
 [ 301  431  871  881  496]
 [ 301  301  311  346  301]
 [ 301  301  301  566  301]
 [ 301  301  536  301 1000]
 [ 301  516  301  931 1000]
 [ 301  301  421 1000 1000]
 [ 426  301 1000 1000 1000]
 [ 301  301  301  456  301]
 [ 301  936 1000 1000 1000]
 [ 301  301  301 1000 1000]
 [ 301  521  301 1000 1000]
 [ 301  301  301 1000  701]
 [ 301  306 1000 1000 1000]
 [ 301  471 1000 1000 1000]
 [ 301  376  506 1000  301]
 [ 301  301  381  301 1000]
 [ 301  301  506  596 1000]
 [ 301  301  301  301 1000]
 [ 301  436  781  931  981]
 [ 301  301  301  301 1000]
 [ 301 1000 1000 1000 1000]
 [ 301  301 1000 1000 1000]
 [ 401  936 1000 1000 1000]
 [ 301  301  306 1000  301]
 [ 301  301  626 1000 1000]
 [ 301  301  706 1000  301]
 [ 301  706 1000 1000 1000]
 [ 301  301 1000 1000 1000]
 [ 511  596 1000 1000  441]
 [ 301  696 1000 1000 1000]
 [ 301  326  641 1000 1000]
 [ 301  326 1000 1000 1000]
 [1000 1000 1000 1000 1000]
 [ 301 1000 1000 1000  716]
 [1000 1000 1000 1000 1000]
 [ 686 1000 1000 1000 1000]
 [ 301  956 1000 1000 1000]
 [ 301 1000 1000 1000  451]
 [ 301  886 1000 1000 1000]
 [1000 1000 1000 1000 1000]
 [ 591 1000 1000 1000  301]
 [ 301  301 1000 1000 1000]
 [ 691 1000 1000 1000 1000]
 [ 301 1000 1000 1000 1000]
 [ 531 1000 1000 1000  301]
 [1000 1000 1000 1000 1000]
 [ 301 1000 1000 1000 1000]]
 ```