import os

from tools.robot_data_binding.robot_reader import URDFReader

# R1 机器人 URDF(路径相对本文件,便于任意目录运行)
_HERE = os.path.dirname(os.path.abspath(__file__))
URDF_PATH = os.path.join(_HERE, "urdf", "R1", "urdf", "r1_v2_1_0.urdf")


def main():
    reader = URDFReader(URDF_PATH)
    robot = reader.load()

    # 可视化(需要图形界面;无头环境会失败,可注释掉)
    # reader.load_meshes = True
    reader.visualize()


if __name__ == "__main__":
    main()
