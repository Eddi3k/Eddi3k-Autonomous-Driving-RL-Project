# Remember to code your baseline for this problem


# We can tell the model to increment the speed where there are not cars in sight, and to reduce it where there are cars nearby.
# Moreover, we can tell the model to change lanes when there are cars in front of it and no cars in the chosen lane. The favourite lane to do to is the rightmost one, which permits to gain a higher reward.


import gymnasium
import highway_env
import matplotlib.pyplot as plt
import numpy as np


SAFE_DISTANCE = 0.12
SAFE_LATERAL_DISTANCE = 0.05    # Computed more or less considering the length of the vehicles and some safety margin
TOLERANCE = 0.03        # Tolerance considered in the measurement of distances

returns = []

env_name = "highway-v0"
lanes_count = 3
env = gymnasium.make(env_name,
                     config={"lanes_count": lanes_count,
                             "ego_spacing": 1.5,
                             "observation": {
                                "type": "Kinematics",
                                "features": ["presence", "x", "y", "vx", "vy"],
                                "absolute": False,                                          # With 'absolute=False', the coordinates are relative to the ego vehicle
                                "order": "sorted"
                                },
                             "action": {
                                "type": "DiscreteMetaAction"}
                            },
                     render_mode='human')



env.reset()
done, truncated = False, False

episode = 1
episode_steps = 0
episode_return = 0

ACTIONS_ALL = env.unwrapped.action_type.ACTIONS_ALL # type: ignore
ACTIONS_ALL_INV = {v: k for k, v in ACTIONS_ALL.items()}

# We don't know in which lane the ego vehicle is at the beginning of an episode
# We will assume that if current_lane = 0, the lane is the rightmost one, while the value increases of 1 for each upper position the ego-vehicle is in
current_lane = 0

while episode <= 10:
    episode_steps += 1

    action = ACTIONS_ALL_INV["FASTER"]  # Default action is to go faster
    #action = ACTIONS_ALL_INV["IDLE"]
    
    lane_left = ACTIONS_ALL_INV["LANE_LEFT"]
    lane_right = ACTIONS_ALL_INV["LANE_RIGHT"]
    change_to_right = True  # Default value, will be changed later if needed
    change_to_left = True  # Default value, will be changed later if needed

    car_ahead = False
    car_ahead_left_lane = False
    car_ahead_right_lane = False

    if episode_steps == 1:
        action = ACTIONS_ALL_INV["SLOWER"]  # Start going slower to avoid some collisions
    elif episode_steps < lanes_count + 1:   # "+1" as the first action we do is going slower
        action = ACTIONS_ALL_INV["LANE_RIGHT"]  # We first "push" the car to the rightmost lane, which permits to get more reward
    else:
        for i in range(0, len(obs)-1):
            if current_lane == 0: car_ahead_right_lane = True  # If the ego vehicle is in the rightmost lane, there is no car ahead in the right lane
            elif current_lane == lanes_count - 1: car_ahead_left_lane = True

            if obs[i][0] == 1 and obs[i][1] > -SAFE_DISTANCE and obs[i][1] < SAFE_DISTANCE:  # Car ahead of ego-vehicle detected

                if obs[i][2] < TOLERANCE and obs[i][2] > -TOLERANCE:
                    car_ahead = True
                    #print("Car ahead detected!")
                elif obs[i][2] < - 1 / lanes_count + TOLERANCE and obs[i][2] > - 2 / lanes_count - TOLERANCE:
                    car_ahead_left_lane = True
                    #print("Car ahead in left lane detected!")
                elif obs[i][2] > 1 / lanes_count - TOLERANCE and obs[i][2] < 2 / lanes_count + TOLERANCE:
                    car_ahead_right_lane = True
                    #print("Car ahead in right lane detected!")

        if episode_steps < lanes_count and not car_ahead_right_lane:
            action = ACTIONS_ALL_INV["LANE_RIGHT"]  # We first "push" the car to the rightmost lane (if the right lane is free), which permits to get more reward
        elif not car_ahead:
            action = ACTIONS_ALL_INV["FASTER"]  # Go faster if no car is ahead
        elif car_ahead and not car_ahead_right_lane:
            action = ACTIONS_ALL_INV["LANE_RIGHT"]  # Change to right lane if it's free (we prefer this with respect to changing to left lane, as being in the rightmost lane gives more reward)
            current_lane -= 1   # We need to distinct this case from the first one of the if-statement in order to properly update the current_lane variable
        elif car_ahead and not car_ahead_left_lane:
            action = ACTIONS_ALL_INV["LANE_LEFT"]  # Change to left lane if it's free
            current_lane += 1
        elif car_ahead and car_ahead_left_lane and car_ahead_right_lane:
            action = ACTIONS_ALL_INV["SLOWER"]  # Go slower if cars are detected in all lanes

    # Reset the flags which indicates if there are vehicles ahead of the ego one for the next step
    car_ahead = False
    car_ahead_left_lane = False
    car_ahead_right_lane = False

    env.render()
    obs, reward, done, truncated, _ = env.step(action)

    # Change the SAFE_DISTANCE parameter based on the ego_vehicle speed (if the speed is high, we will need a higher safe distance in order not to crash)
    if obs[0][3] > 0.36:
        SAFE_DISTANCE = 0.26
    elif obs[0][3] > 0.31:
        SAFE_DISTANCE = 0.23
    elif SAFE_DISTANCE > 0.14:
        SAFE_DISTANCE = 0.14

    episode_return += float(reward)

    if done or truncated:
        print(f"Episode Num: {episode} Episode T: {episode_steps} Return: {episode_return:.3f}, Crash: {done}")
        returns.append(episode_return)

        env.reset()
        episode += 1
        episode_steps = 0
        episode_return = 0
        print("------------------------------------------------------------------------------------")
env.close()



main_return = np.mean(returns)
main_returns = [main_return] * len(returns)

plt.plot(returns)
plt.plot(main_returns)
plt.xlabel("Episode")
plt.ylabel("Return")
plt.title("Episode Returns")
plt.show()