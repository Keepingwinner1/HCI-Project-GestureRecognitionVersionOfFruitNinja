# Fruit Ninja 游戏

  这是一个**基于Pygame、Mediapipe和OpenCV**实现的**PC端游戏**。玩家通过手势控制切割屏幕上的水果，尽可能多地获取分数。在原有游戏的基础上，我们**创新地引入了手势识别与微笑识别，使得交互更加有趣和好玩**。

## 目录

- [Fruit Ninja 游戏](#fruit-ninja-游戏)
  - [目录](#目录)
  - [安装](#安装)
  - [运行游戏](#运行游戏)
  - [游戏说明](#游戏说明)
  - [文件目录结构](#文件目录结构)

## 安装

1. 创建虚拟环境并激活:
   
   ```
   conda create -n FruitNinja
   conda activate FruitNinja
   ```

2. 安装环境依赖
   
   ```
   pip install -r requirements.txt
   ```

3. 使用VScode打开本项目文件夹 

4. 选择创建好的虚拟环境FruitNinja

## 运行游戏

在运行代码之前请确保你已经连接了一个网络摄像头并正确安装了所有依赖项。

```
python main.py
```

## 游戏说明

- 目标: 通过手势在屏幕上切割水果获取分数。
- 计时: 游戏时长为30秒，倒计时结束后显示最终得分。
- 操作: 将手放在摄像头前，通过移动食指来切割水果。
- 控制：
  - 开始游戏: 启动程序后，保持微笑，即可进入游戏，倒计时开始，开始切割水果。
  - 重新开始游戏: 游戏结束后，点击屏幕上的重新开始按钮重新开始游戏。

## 文件目录结构

```
fruit-ninja-game/
│
├── images/
│   ├── apple.png
│   ├── apple-1.png
│   ├── apple-2.png
│   ├── banana.png
│   ├── banana-1.png
│   ├── banana-2.png
│   ├── peach.png
│   ├── peach-1.png
│   ├── peach-2.png
│   ├── strawberry.png
│   ├── strawberry-1.png
│   ├── strawberry-2.png
│   ├── watermelon.png
│   ├── watermelon-1.png
│   ├── watermelon-2.png
│   ├── logo.png
│   ├── new-game.png
│   ├── new-game-hover.png
│   ├── new-game-click.png
│
├── sounds/
│   ├── bg.mp3
│   └── cut.mp3
│
├── main.py
├── requirements.txt
├── README.md
└── .gitignore
```
