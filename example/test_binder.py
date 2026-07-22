import glob
import os

from tools.robot_data_binder.robot_data_binder import RobotDataBinder

_HERE = os.path.dirname(os.path.abspath(__file__))
URDF_PATH = os.path.join(_HERE, "urdf", "R1", "urdf", "r1_v2_1_0.urdf")
H5_PATH = sorted(glob.glob(os.path.join(_HERE, "data", "*", "*.h5")))[0]

# R1 手臂关节名(取自数据 metadata 的 joint_names),按顺序对应 arm 数组的列
_ARM_JOINTS = [
    "left_arm_joint1", "left_arm_joint2", "left_arm_joint3",
    "left_arm_joint4", "left_arm_joint5", "left_arm_joint6",
    "right_arm_joint1", "right_arm_joint2", "right_arm_joint3",
    "right_arm_joint4", "right_arm_joint5", "right_arm_joint6",
]


def main():
    binder = RobotDataBinder(URDF_PATH, H5_PATH)
    binder.load()

    # 1) 建立映射
    # 方式 A:交互式逐关节选择(先列出数据候选,再逐个关节输入序号/列号)
    binder.binding()

    # 方式 B:程序化直接给映射(每个关节可绑 state 和 action 两路,各是 (数据路径, 列号))
    # mapping = {
    #     name: {
    #         "state": ("joints/state/arm/position", col),
    #         "action": ("joints/action/arm/position", col),
    #     }
    #     for col, name in enumerate(_ARM_JOINTS)
    # }
    # binder.binding(mapping=mapping)

    # 2) 指定相机(可多个)。方式 A:交互式(输入相机 key 名 + 选数据);方式 B:程序化给 {key: 数据路径}
    binder.bind_camera()
    # binder.bind_camera(cameras={
    #     "head": "cameras/head/color/data",
    #     "hand_left": "cameras/hand_left/color/data",
    # })

    # 2.5) 保存/恢复映射关系(关节绑定 + 相机)
    binder.save_binding(os.path.join(_HERE, "binding.json"))
    # 之后可在新 binder 上一步恢复,免去重新交互:
    # binder.load_binding(os.path.join(_HERE, "binding.json"))

    # 3) 回放(需要图形界面,按需打开)
    # 3a) 只驱动机器人
    # binder.replay()
    # 3b) 驱动机器人 + 同步播放指定相机视频(需先 bind_camera)
    binder.replay_with_camera()


if __name__ == "__main__":
    main()
