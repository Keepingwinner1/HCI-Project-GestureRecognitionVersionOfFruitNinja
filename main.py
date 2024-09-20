import pygame
import cv2
import mediapipe as mp
import random
import sys
from scipy.interpolate import splprep, splev
import numpy as np

# 初始化pygame
pygame.init()
pygame.mixer.init()
screen_width, screen_height = 1000, 800
screen = pygame.display.set_mode((screen_width, screen_height))
pygame.display.set_caption("Fruit Ninja")

# 加载摄像头
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open camera.")
    sys.exit()

# 加载手部检测模型
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_draw = mp.solutions.drawing_utils
# 加载预训练的人脸、笑脸检测模型
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
smile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_smile.xml')

# 物理参数
GRAVITY = 0.7  # 重力加速度
AIR_RESISTANCE = 0.05  # 空气阻力


def draw_smooth_line(screen, points, color, width):
    if len(points) < 2:
        return
    
    s = pygame.Surface(screen.get_size(), pygame.SRCALPHA)  # 支持透明度
    for i in range(len(points) - 1):
        start_pos = points[i]
        end_pos = points[i + 1]
        pygame.draw.line(s, color, start_pos, end_pos, width)
    screen.blit(s, (0, 0))

def resize_and_top_align_image(img, window_width, window_height):
    """将图片缩放并顶部对齐显示在固定窗口大小内"""
    img_height, img_width = img.shape[:2]
    scale = min(window_width / img_width, window_height / img_height)
    new_width = int(img_width * scale)
    new_height = int(img_height * scale)
    resized_img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_AREA)

    # 创建一个全透明的背景图像（具有4个通道）
    background = np.zeros((window_height, window_width, 4), dtype=np.uint8)

    # 计算水平居中的位置，垂直方向对齐顶部
    x_offset = (window_width - new_width) // 2
    y_offset = 0  # 顶部对齐

    # 将缩放后的图像粘贴到背景图像上
    background[y_offset:y_offset+new_height, x_offset:x_offset+new_width] = resized_img

    return background, (x_offset, y_offset, new_width, new_height)

def overlay_image_alpha(background, overlay, x, y):
    """将带有alpha通道的叠加图像覆盖到背景图像上"""
    b, g, r, a = cv2.split(overlay)
    overlay_color = cv2.merge((b, g, r))

    mask = a / 255.0
    for c in range(0, 3):
        background[y:y+overlay.shape[0], x:x+overlay.shape[1], c] = (background[y:y+overlay.shape[0], x:x+overlay.shape[1], c] * (1 - mask) + overlay_color[:, :, c] * mask).astype(np.uint8)

# 定义水果类
class Fruit:
    def __init__(self, image, x, y):
        self.image = image
        self.rect = self.image.get_rect(center=(x, y))
        self.is_cut = False
        self.velocity_x = random.uniform(-30, 30)  # 初始速度
        self.velocity_y = random.uniform(-60, -45)  # 初始速度
        self.rotation_angle = 0  # 初始旋转角度
        self.rotation_speed = random.uniform(1, 3)  # 随机旋转速度

    def reset(self):
        self.rect.x = random.randint(100, screen_width - 100)
        self.rect.y = screen_height + 100
        self.velocity_x = random.uniform(-30, 30)
        self.velocity_y = random.uniform(-60, -45)
        self.rotation_angle = 0
        self.rotation_speed = random.uniform(1, 3)
        self.is_cut = False

    def update(self):
        if not self.is_cut:
            # 应用重力
            self.velocity_y += GRAVITY
            # 应用空气阻力
            self.velocity_x *= (1 - AIR_RESISTANCE)
            self.velocity_y *= (1 - AIR_RESISTANCE)
            # 更新位置
            self.rect.x += self.velocity_x
            self.rect.y += self.velocity_y
            # 更新旋转角度
            self.rotation_angle += self.rotation_speed
            if self.rotation_angle > 360:
                self.rotation_angle -= 360
            # 检查是否掉落出屏幕
            if self.rect.y > screen_height:
                self.reset()

    def draw(self, screen):
        rotated_image = pygame.transform.rotozoom(self.image, self.rotation_angle, 1)  # 使用rotozoom平滑旋转
        rotated_rect = rotated_image.get_rect(center=self.rect.center)
        screen.blit(rotated_image, rotated_rect.topleft)

    def check_cut(self, finger_positions):
        if len(finger_positions) < 2:
            return False
        
        for i in range(len(finger_positions) - 1):
            start_pos = finger_positions[i]
            end_pos = finger_positions[i + 1]
            if self.rect.clipline(start_pos, end_pos):
                distance = pygame.math.Vector2(start_pos).distance_to(end_pos)
                if distance > self.rect.width:
                    return True
        return False

class HalfFruit:
    def __init__(self, image, x, y, vx, vy, angle):
        self.image = image
        self.x = x
        self.y = y
        self.vx = vx
        self.vy = vy
        self.angle = angle
        self.gravity = GRAVITY

    def update(self):
        self.x += self.vx
        self.y += self.vy
        self.vy += self.gravity
        self.angle += 5

    def draw(self, screen):
        rotated_image = pygame.transform.rotate(self.image, self.angle)
        screen.blit(rotated_image, (self.x, self.y))


class HandDetector:
    def __init__(self, max_num_hands=1):
        self.hands = mp.solutions.hands.Hands(max_num_hands=max_num_hands)
        self.mp_draw = mp.solutions.drawing_utils
        self.hand_landmarks = None

    def detect_hands(self, frame):
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self.hand_landmarks = self.hands.process(frame_rgb).multi_hand_landmarks
        return self.hand_landmarks

    def draw_hands(self, frame):
        if self.hand_landmarks:
            for hand_landmarks in self.hand_landmarks:
                self.mp_draw.draw_landmarks(frame, hand_landmarks, mp.solutions.hands.HAND_CONNECTIONS)

    def get_index_finger_tip_position(self, frame_width, frame_height):
        if self.hand_landmarks:
            for hand_landmarks in self.hand_landmarks:
                index_finger_tip = hand_landmarks.landmark[mp.solutions.hands.HandLandmark.INDEX_FINGER_TIP]
                return int(index_finger_tip.x * frame_width), int(index_finger_tip.y * frame_height)
        return None, None
    

# 初始化pygame音效
pygame.mixer.init()
cut_sound = pygame.mixer.Sound('./sound/cut.mp3')
background_music = pygame.mixer.Sound('./sound/bg.mp3')
BGM_Start = pygame.mixer.Sound('./sound/happyfly.ogg')  

# 加载logo图片
logo_image = pygame.image.load('./image/logo.png').convert_alpha()
# 加载图片素材
restart_button_image = pygame.image.load('./image/new-game.png')
restart_button_hover = pygame.image.load('./image/new-game-hover.png')
restart_button_click = pygame.image.load('./image/new-game-click.png')
restart_button_click = pygame.transform.scale(restart_button_click, (200, 200))
restart_button_image = pygame.transform.scale(restart_button_image, (200, 200))  # 调整按钮大小
restart_button_hover = pygame.transform.scale(restart_button_hover, (200, 200))
# 加载水果图片
try:
    fruit_images = [
        pygame.transform.scale(pygame.image.load('./image/apple.png'), (150, 150)),   # 苹果
        pygame.transform.scale(pygame.image.load('./image/banana.png'), (155, 75)),  # 香蕉
        pygame.transform.scale(pygame.image.load('./image/peach.png'), (150, 150)),   # 桃子
        pygame.transform.scale(pygame.image.load('./image/watermelon.png'), (200, 200)),  # 西瓜
        pygame.transform.scale(pygame.image.load('./image/strawberry.png'), (160, 160))  # 草莓
    ]

    fruit_left_images = [
        pygame.transform.scale(pygame.image.load('./image/apple-1.png'), (150, 150)),   # 苹果
        pygame.transform.scale(pygame.image.load('./image/banana-1.png'), (155, 75)),  # 香蕉
        pygame.transform.scale(pygame.image.load('./image/peach-1.png'), (150, 150)),   # 桃子
        pygame.transform.scale(pygame.image.load('./image/watermelon-1.png'), (200, 200)),  # 西瓜
        pygame.transform.scale(pygame.image.load('./image/strawberry-1.png'), (160, 160))  # 草莓
    ]
    

    fruit_right_images = [
        pygame.transform.scale(pygame.image.load('./image/apple-2.png'), (150, 150)),   # 苹果
        pygame.transform.scale(pygame.image.load('./image/banana-2.png'), (155, 75)),  # 香蕉
        pygame.transform.scale(pygame.image.load('./image/peach-2.png'), (150, 150)),   # 桃子
        pygame.transform.scale(pygame.image.load('./image/watermelon-2.png'), (200, 200)),  # 西瓜
        pygame.transform.scale(pygame.image.load('./image/strawberry-2.png'), (160, 160))  # 草莓
    ]

except pygame.error as e:
    print(f"Error loading images: {e}")
    sys.exit()


# 创建水果实例
fruits = []
half_fruits = []
fruit_spawn_timer = 0  # 计时器，控制水果抛出间隔
fruit_spawn_interval = 80  # 抛出水果的间隔帧数

# 初始化手势检测
hand_detector = HandDetector()
index_finger_positions = []

# 游戏主循环
smile_detected = False
running = True
score = 0
clock = pygame.time.Clock()
game_duration = 30  # 游戏时长 30 秒
game_over = False
start_time = pygame.time.get_ticks()
showText =False

while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False


    if game_over:
        # 显示摄像头画面
        ret, frame = cap.read()
        if ret:
            frame = cv2.flip(frame, 1)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_surface = pygame.surfarray.make_surface(frame_rgb)
            frame_surface = pygame.transform.rotate(frame_surface, -90)
            frame_surface = pygame.transform.flip(frame_surface, True, False)
            frame_surface = pygame.transform.scale(frame_surface, (screen_width, screen_height))
            screen.blit(frame_surface, (0, 0))

        font = pygame.font.SysFont(None, 55)
        # 显示最终得分
        final_score_text = font.render(f'Final Score: {score}', True, (255, 255, 255))
        final_score_rect = final_score_text.get_rect(center=(screen_width // 2, screen_height // 2 - 50))
        screen.blit(final_score_text, final_score_rect)

        # 显示重新开始按钮
        button_x = screen_width // 2 - 200 // 2
        button_y = screen_height // 2 + 50
        mouse_pos = pygame.mouse.get_pos()
        mouse_pressed = pygame.mouse.get_pressed()

        if restart_button_image.get_rect(topleft=(button_x, button_y)).collidepoint(mouse_pos):
            if mouse_pressed[0]:  # 左键按下
                screen.blit(restart_button_click, (button_x, button_y))
            else:
                screen.blit(restart_button_hover, (button_x, button_y))
        else:
            screen.blit(restart_button_image, (button_x, button_y))

        restart_rect = pygame.Rect(button_x, button_y, 200, 200)

        pygame.display.flip()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if restart_rect.collidepoint(event.pos):
                    # 重置游戏
                    start_time = pygame.time.get_ticks()
                    game_over = False
                    score = 0
                    fruits = []
                    half_fruits = []
                    index_finger_positions = []
        continue

    # 捕获视频帧
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame.")
        break
    # 翻转摄像头图像
    frame = cv2.flip(frame, 1)

    if not smile_detected:
            BGM_Start.play(loops=-1) 
            # 显示标题画面
            title_image = cv2.imread('./image/title.png', cv2.IMREAD_UNCHANGED)
            # 调整并顶部对齐图片，获取其尺寸和位置
            top_aligned_image, (x_offset, y_offset, new_width, new_height) = resize_and_top_align_image(title_image, screen_width, screen_height)
            # 将摄像头帧调整为窗口大小
            frame_resized = cv2.resize(frame, (screen_width, screen_height))
            # 将顶部对齐图片叠加到摄像头背景上
            overlay_image_alpha(frame_resized, top_aligned_image, x_offset, y_offset)
            # 将OpenCV图像转换为RGB格式
            frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
            # 将OpenCV图像转换为Pygame图像
            frame_surface = pygame.surfarray.make_surface(frame_rgb)
            # 旋转并翻转图像
            frame_surface = pygame.transform.rotate(frame_surface, -90)
            frame_surface = pygame.transform.flip(frame_surface, True, False)
            # 将图像绘制到屏幕上
            screen.blit(frame_surface, (0, 0))
            # 笑脸检测逻辑
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.3, 5)
            for (x, y, w, h) in faces:
                # 获取人脸的灰度图像
                roi_gray = gray[y:y+h, x:x+w]
                # 在人脸区域内检测笑脸
                smiles = smile_cascade.detectMultiScale(roi_gray, 1.6, 15)
                if len(smiles) > 0:
                    # 检测到笑脸，进入游戏
                    smile_detected = True
                    # 显示笑脸检测成功提示
                    print("Smile detected!")
                    chinese_font = pygame.font.Font('xiangjao.ttf', 80)
                    message_text = chinese_font.render('游戏将在5秒后开始,伸出你的手~', True, (255, 255, 255))
                    message_rect = message_text.get_rect(center=(screen_width // 2, screen_height // 2))
                    showText = True
                    # 记录文本显示的开始时间
                    message_display_time = pygame.time.get_ticks()
                    break
    elif showText:
        current_time = pygame.time.get_ticks()
        hand_landmarks = hand_detector.detect_hands(frame)
        hand_detector.draw_hands(frame)
        frame_resized = cv2.resize(frame, (screen_width, screen_height))
        # 将OpenCV图像转换为RGB格式
        frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
        # 将OpenCV图像转换为Pygame图像
        frame_surface = pygame.surfarray.make_surface(frame_rgb)
        # 旋转并翻转图像
        frame_surface = pygame.transform.rotate(frame_surface, -90)
        frame_surface = pygame.transform.flip(frame_surface, True, False)
        # 将图像绘制到屏幕上
        screen.blit(frame_surface, (0, 0))
        # 显示文本
        if current_time - message_display_time < 5000:  
            screen.blit(message_text, message_rect)
        else:
            showText =False
            start_time = pygame.time.get_ticks()
    else:
        BGM_Start.stop()
        background_music.play(loops=-1)  # 循环播放背景音乐
        hand_landmarks = hand_detector.detect_hands(frame)
        hand_detector.draw_hands(frame)
        # 将OpenCV图像转换为RGB格式
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # 将OpenCV图像转换为Pygame图像
        frame_surface = pygame.surfarray.make_surface(frame_rgb)
        # 旋转并翻转图像
        frame_surface = pygame.transform.rotate(frame_surface, -90)
        frame_surface = pygame.transform.flip(frame_surface, True, False)
        hand_x, hand_y = hand_detector.get_index_finger_tip_position(screen_width, screen_height)
        if hand_x and hand_y:
            index_finger_positions.append((hand_x, hand_y))
            if len(index_finger_positions) > 15:
                index_finger_positions.pop(0)

        if len(index_finger_positions) > 1:
            for fruit in fruits:
                if fruit.check_cut(index_finger_positions):
                    fruit.is_cut = True
                    score += 1
                    cut_sound.play()
                    fruit_index = fruit_images.index(fruit.image)
                    half_fruits.append(HalfFruit(fruit_left_images[fruit_index], fruit.rect.x, fruit.rect.y, fruit.velocity_x - 1, fruit.velocity_y, -45))
                    half_fruits.append(HalfFruit(fruit_right_images[fruit_index], fruit.rect.x + 50, fruit.rect.y, fruit.velocity_x + 1, fruit.velocity_y, 45))
                    fruit.reset()
                    # 清空位置列表以准备下一次切割
                    index_finger_positions.clear()


        # 更新水果位置
        for fruit in fruits:
            fruit.update()

        for half_fruit in half_fruits:
            half_fruit.update()

        half_fruits = [hf for hf in half_fruits if hf.y <= screen_height]

        # 更新水果生成计时器
        # 生成新的水果
        min_fruits_per_throw = 1
        max_fruits_per_throw = 3
        fruit_spawn_timer += 1
        if fruit_spawn_timer >= fruit_spawn_interval:
            num_fruits = random.randint(min_fruits_per_throw, max_fruits_per_throw)
            for _ in range(num_fruits):
                fruit_image = random.choice(fruit_images)
                fruits.append(Fruit(fruit_image, random.randint(100, screen_width - 100), screen_height))
            fruit_spawn_timer = 0

        # 绘制背景和水果
        screen.fill((255, 255, 255))
        screen.blit(pygame.transform.scale(frame_surface, (screen_width, screen_height)), (0, 0))
        for fruit in fruits:
            fruit.draw(screen)
        for half_fruit in half_fruits:
            half_fruit.draw(screen)

        # 显示手势轨迹
        # 绘制手势轨迹，使用平滑线条绘制
        if len(index_finger_positions) > 1:
            draw_smooth_line(screen, index_finger_positions, (255, 255, 255, 128), 5)

        # 计算剩余时间
        elapsed_time = (pygame.time.get_ticks() - start_time) // 1000
        remaining_time = max(0, game_duration - elapsed_time)

        font = pygame.font.SysFont(None, 55)
        # 显示倒计时
        time_text = font.render(f'Time: {remaining_time}', True, (0, 0, 0))
        screen.blit(time_text, (screen_width - 150, 10))

        # 检查游戏是否结束
        if remaining_time == 0:
            game_over = True

        # 显示分数
        score_text = font.render(f'Score: {score}', True, (0, 0, 0))
        screen.blit(score_text, (10, 10))

        # 绘制 logo
        screen.blit(logo_image, (screen_width // 2 - logo_image.get_width() // 2, 10))

    pygame.display.flip()
    clock.tick(30)

cap.release()
cv2.destroyAllWindows()
pygame.quit()



