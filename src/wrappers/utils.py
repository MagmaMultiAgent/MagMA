import numpy as np
from kit.config import EnvConfig
from luxai_s2.unit import FactoryPlacementActionType, BidActionType
from luxai_s2.state import ObservationStateDict
from scipy.spatial import KDTree
from scipy.ndimage import distance_transform_cdt

ICE_WEIGHTS = np.array([1, 0.5, 0.33, 0.25]) 
ORE_WEIGHTS = np.array([1, 0.5, 0.33, 0.25])
ICE_PREF = 3

def manhattan_dist_to_nth_closest(arr, n):
    if n == 1:
        distance_map = distance_transform_cdt(1-arr, metric='taxicab')
        return distance_map
    else:
        true_coords = np.transpose(np.nonzero(arr))
        tree = KDTree(true_coords)
        dist, _ = tree.query(np.transpose(np.nonzero(~arr)), k=n, p=1)
        return np.reshape(dist[:, n-1], arr.shape)
    
def count_region_cells(array, start, min_dist=2, max_dist=np.inf, exponent=1):
    
    def dfs(array, loc):
        distance_from_start = abs(loc[0]-start[0]) + abs(loc[1]-start[1])
        if not (0<=loc[0]<array.shape[0] and 0<=loc[1]<array.shape[1]):
            return 0
        if (not array[loc]) or visited[loc]:
            return 0
        if not (min_dist <= distance_from_start <= max_dist):      
            return 0
        
        visited[loc] = True

        count = 1.0 * exponent**distance_from_start
        count += dfs(array, (loc[0]-1, loc[1]))
        count += dfs(array, (loc[0]+1, loc[1]))
        count += dfs(array, (loc[0], loc[1]-1))
        count += dfs(array, (loc[0], loc[1]+1))
        return count

    visited = np.zeros_like(array, dtype=bool)
    return dfs(array, start)

def factory_placement(player, obs: ObservationStateDict) -> FactoryPlacementActionType:

    ice_distances = [manhattan_dist_to_nth_closest(obs["board"]["ice"], i) for i in range(1, 5)]
    ore_distances = [manhattan_dist_to_nth_closest(obs["board"]["ore"], i) for i in range(1, 5)]
    weigthed_ice_dist = np.sum(np.array(ice_distances) * ICE_WEIGHTS[:, np.newaxis, np.newaxis], axis=0)
    weigthed_ore_dist = np.sum(np.array(ore_distances) * ORE_WEIGHTS[:, np.newaxis, np.newaxis], axis=0)
    combined_resource_score = (weigthed_ice_dist * ICE_PREF + weigthed_ore_dist)
    combined_resource_score = (np.max(combined_resource_score) - combined_resource_score) * obs["board"]["valid_spawns_mask"]
    rubble = obs["board"]["rubble"]
    low_rubble = (rubble < 25)
    low_rubble_scores = np.zeros_like(low_rubble, dtype=np.float32)
    for i in range(low_rubble.shape[0]):
        for j in range(low_rubble.shape[1]):
            low_rubble_scores[i,j] = count_region_cells(low_rubble, (i,j), min_dist=0, max_dist=8, exponent=0.9)

    overall_score = (low_rubble_scores*2 + combined_resource_score ) * obs["board"]["valid_spawns_mask"]
    best_loc = np.argmax(overall_score)
    pos = np.unravel_index(best_loc, overall_score.shape)

    metal = obs["teams"][player]["metal"]
    water = obs["teams"][player]["water"]
    return dict(spawn=pos, metal=int(metal/EnvConfig.MIN_FACTORIES), water=int(water/EnvConfig.MIN_FACTORIES))

BASE_BID = 0
MAX_BID = 25
BIAS_FACTOR = 2

def bid_with_log_bias(player, obs: ObservationStateDict) -> BidActionType:

    random_value = np.random.rand()
    log_bias = -np.log(random_value) / BIAS_FACTOR

    bid = BASE_BID + (MAX_BID - BASE_BID) * log_bias
    bid = min(MAX_BID, max(BASE_BID, bid))

    faction = "AlphaStrike"
    if player == "player_1":
        faction = "MotherMars"

    return dict(bid=bid, faction=faction)

