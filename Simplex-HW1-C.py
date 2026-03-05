"""
Задача C*: Визуализация симплекс-метода в 3D

Модельная задача:
Фабрика производит столы (x₁), стулья (x₂) и шкафы (x₃).
Нужно максимизировать прибыль при ограничениях на ресурсы.

Целевая функция: F = 8x₁ + 5x₂ + 10x₃ → max

Ограничения:
1) x₁ + 2x₂ + 3x₃ ≤ 180  (древесина)
2) 2x₁ + x₂ + 3x₃ ≤ 200  (труд)
3) x₃ ≥ 20               (минимум шкафов)
4) x₁ + x₂ ≥ 30          (минимум столов и стульев)
5) x₁, x₂, x₃ ≥ 0

Вершины многогранника (найдены симплекс-методом):
A(30,0,20)  → B(70,0,20) → C(60,20,20) → D(53.33,33.33,20)
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import animation
import matplotlib.patches as mpatches

# -------------------------------------------------------------------
# 1. Исходные данные: вершины и значения целевой функции
# -------------------------------------------------------------------

# Координаты вершин в порядке обхода симплекс-методом
# Каждая строка: [x₁, x₂, x₃]
V = np.array([
    [30, 0, 20],        # Вершина A: начальная после Фазы I
    [70, 0, 20],        # Вершина B: после первой итерации
    [60, 20, 20],       # Вершина C: промежуточная
    [160/3, 100/3, 20]  # Вершина D: оптимальная (53.33, 33.33, 20)
])

# Имена вершин для подписей
V_NAMES = ['A (30,0,20)', 'B (70,0,20)', 'C (60,20,20)', 'D* (53.3,33.3,20)']

# Цвета для разных вершин (чтобы их можно было различать)
V_COLORS = ['#FF4444', '#FF8844', '#44FF44', '#4444FF']

# Значения целевой функции в каждой вершине
F = np.array([8, 5, 10]) @ V.T  # матричное умножение: (8,5,10) * V^T

print('Координаты вершин и значения F:')
for i in range(len(V)):
    print(f'  {V_NAMES[i]}: F = {F[i]:.2f}')

# -------------------------------------------------------------------
# 2. Настройка трёхмерного графика
# -------------------------------------------------------------------

# Создаём фигуру с подходящим размером
fig = plt.figure(figsize=(14, 10))
ax = fig.add_subplot(111, projection='3d')

# Подписи осей с указанием размерности
ax.set_xlabel('x₁ (столы), шт.', fontsize=11, labelpad=10)
ax.set_ylabel('x₂ (стулья), шт.', fontsize=11, labelpad=10)
ax.set_zlabel('x₃ (шкафы), шт.', fontsize=11, labelpad=10)

# Заголовок
ax.set_title('Движение симплекс-метода по вершинам допустимого многогранника', 
             fontsize=14, pad=20)

# Границы графика (с небольшим запасом)
ax.set_xlim(0, 100)
ax.set_ylim(0, 60)
ax.set_zlim(15, 45)

# Сетка для лучшего восприятия глубины
ax.grid(True, alpha=0.3, linestyle=':')

# -------------------------------------------------------------------
# 3. Статические элементы: вершины и рёбра многогранника
# -------------------------------------------------------------------

# Отмечаем все вершины (они будут видны всегда, не только в анимации)
for i, (coord, name, color) in enumerate(zip(V, V_NAMES, V_COLORS)):
    ax.scatter(*coord, color=color, s=150, edgecolor='black', linewidth=2, alpha=0.8)
    ax.text(coord[0], coord[1], coord[2]+2, name, 
            fontsize=9, ha='center', va='bottom',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.9))

# Рисуем рёбра многогранника (соединяем вершины в порядке обхода)
# Используем пунктир, чтобы не перегружать график
for i in range(len(V)-1):
    ax.plot([V[i,0], V[i+1,0]], 
            [V[i,1], V[i+1,1]], 
            [V[i,2], V[i+1,2]], 
            color='gray', linestyle='--', linewidth=1.5, alpha=0.6)

# -------------------------------------------------------------------
# 4. Плоскости ограничений (полупрозрачные)
# -------------------------------------------------------------------

# Сетка для построения плоскостей
x = np.linspace(0, 100, 25)
y = np.linspace(0, 60, 25)
X, Y = np.meshgrid(x, y)

# Плоскость 1: x₁ + 2x₂ + 3x₃ = 180  →  x₃ = (180 - x₁ - 2x₂)/3
Z1 = (180 - X - 2*Y) / 3
# Оставляем только ту часть, где x₃ ≥ 20 и x₁ + x₂ ≥ 30
Z1 = np.where((Z1 >= 20) & (X + Y >= 30), Z1, np.nan)

# Плоскость 2: 2x₁ + x₂ + 3x₃ = 200  →  x₃ = (200 - 2x₁ - x₂)/3
Z2 = (200 - 2*X - Y) / 3
Z2 = np.where((Z2 >= 20) & (X + Y >= 30), Z2, np.nan)

# Плоскость 3: x₃ = 20 (нижняя граница по шкафам)
Z3 = np.full_like(X, 20)
# Оставляем только там, где выполняются ресурсные ограничения
mask3 = (X + 2*Y + 3*Z3 <= 180) & (2*X + Y + 3*Z3 <= 200) & (X + Y >= 30)
X3 = np.where(mask3, X, np.nan)
Y3 = np.where(mask3, Y, np.nan)

# Плоскость 4: x₁ + x₂ = 30 (вертикальная плоскость)
# Для вертикальной плоскости нужен другой подход: фиксируем x₂ = 30 - x₁
x_line = np.linspace(0, 80, 20)
z_line = np.linspace(20, 45, 15)
X4, Z4 = np.meshgrid(x_line, z_line)
Y4 = 30 - X4
mask4 = (X4 + 2*Y4 + 3*Z4 <= 180) & (2*X4 + Y4 + 3*Z4 <= 200) & (Z4 >= 20)
X4 = np.where(mask4, X4, np.nan)
Y4 = np.where(mask4, Y4, np.nan)

# Рисуем плоскости с разными цветами и прозрачностью
surf1 = ax.plot_surface(X, Y, Z1, alpha=0.15, color='steelblue', label='_nolegend_')
surf2 = ax.plot_surface(X, Y, Z2, alpha=0.15, color='seagreen', label='_nolegend_')
surf3 = ax.plot_surface(X3, Y3, Z3, alpha=0.2, color='lightcoral', label='_nolegend_')
surf4 = ax.plot_surface(X4, Y4, Z4, alpha=0.15, color='gold', label='_nolegend_')

# -------------------------------------------------------------------
# 5. Элементы, которые будут меняться в анимации
# -------------------------------------------------------------------

# Линия, показывающая пройденный путь (будет удлиняться)
trail, = ax.plot([], [], [], 'o-', color='purple', linewidth=3,
                 markersize=8, markerfacecolor='white', markeredgecolor='purple',
                 label='Пройденный путь')

# Текущая вершина (будет перемещаться)
current, = ax.plot([], [], [], 'o', color='red', markersize=16,
                   markerfacecolor='yellow', markeredgecolor='red', markeredgewidth=3,
                   label='Текущая вершина')

# Текстовое поле с информацией об итерации
info = ax.text2D(0.02, 0.95, '', transform=ax.transAxes, fontsize=11,
                 verticalalignment='top',
                 bbox=dict(boxstyle='round,pad=0.5', facecolor='wheat', alpha=0.95))

# Легенда (помещаем в правый верхний угол)
ax.legend(loc='upper right', fontsize=10)

# -------------------------------------------------------------------
# 6. Функции для анимации
# -------------------------------------------------------------------

def init_animation():
    """Инициализация: очищаем анимируемые элементы"""
    trail.set_data([], [])
    trail.set_3d_properties([])
    current.set_data([], [])
    current.set_3d_properties([])
    info.set_text('')
    return trail, current, info

def animate(frame):
    """
    frame: номер кадра (0,1,2,3)
    Каждый кадр соответствует одной вершине
    """
    idx = frame
    
    # Пройденный путь: все вершины от 0 до idx
    trail.set_data(V[:idx+1, 0], V[:idx+1, 1])
    trail.set_3d_properties(V[:idx+1, 2])
    
    # Текущая вершина
    current.set_data([V[idx, 0]], [V[idx, 1]])
    current.set_3d_properties([V[idx, 2]])
    
    # Формируем текст с информацией
    text = f'Итерация {idx+1} из {len(V)}\n'
    text += f'Вершина: {V_NAMES[idx]}\n'
    text += f'F = {F[idx]:.2f}'
    
    if idx == len(V)-1:
        text += ' ← ОПТИМУМ!'
        current.set_color('gold')  # меняем цвет для оптимума
    else:
        current.set_color('red')
    
    info.set_text(text)
    
    return trail, current, info

# Создаём анимацию с интервалом 2.5 секунды между кадрами
anim = animation.FuncAnimation(fig, animate, init_func=init_animation,
                               frames=len(V), interval=2500,
                               repeat=True, blit=False)

# -------------------------------------------------------------------
# 7. Сохранение (раскомментировать при необходимости)
# -------------------------------------------------------------------

# Для сохранения в GIF нужен pillow: pip install pillow
# anim.save('simplex_path.gif', writer='pillow', fps=0.5, dpi=100)

# Для сохранения в MP4 нужен ffmpeg
# anim.save('simplex_path.mp4', writer='ffmpeg', fps=0.5, dpi=200)

# -------------------------------------------------------------------
# 8. Вспомогательный график: 2D-проекции
# -------------------------------------------------------------------

# Создаём отдельное окно с двумя проекциями для лучшего понимания
fig2, axes = plt.subplots(1, 2, figsize=(14, 5))

# Левая проекция: на плоскость x₁-x₂ (вид сверху)
ax1 = axes[0]

# Сетка для построения границ
x = np.linspace(0, 100, 200)
# При x₃=20 ограничения превращаются в:
y1 = (180 - x - 60) / 2      # x₁ + 2x₂ = 120
y2 = (200 - 2*x - 60)        # 2x₁ + x₂ = 140
y3 = 30 - x                  # x₁ + x₂ = 30 (минимальная граница)

# Закрашиваем допустимую область
y_min = np.maximum(0, y3)
y_max = np.minimum(np.minimum(y1, y2), 50)
ax1.fill_between(x, y_min, y_max, where=(y_min <= y_max) & (x >= 0),
                 alpha=0.3, color='lightgray', label='Допустимая область')

# Рисуем линии ограничений
ax1.plot(x, y1, 'b-', linewidth=2, label='x₁+2x₂=120')
ax1.plot(x, y2, 'g-', linewidth=2, label='2x₁+x₂=140')
ax1.plot(x, y3, 'orange', linestyle='--', linewidth=2, label='x₁+x₂=30')

# Отмечаем вершины
for i, (coord, name, color) in enumerate(zip(V, V_NAMES, V_COLORS)):
    ax1.plot(coord[0], coord[1], 'o', color=color, markersize=10, markeredgecolor='black')
    ax1.annotate(name, (coord[0]+2, coord[1]+1), fontsize=8,
                 bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8))

# Соединяем вершины (путь алгоритма)
for i in range(len(V)-1):
    ax1.plot([V[i,0], V[i+1,0]], [V[i,1], V[i+1,1]], 
             'purple', linestyle='--', linewidth=2, alpha=0.7)

ax1.set_xlabel('x₁ (столы)')
ax1.set_ylabel('x₂ (стулья)')
ax1.set_title('Проекция на плоскость x₁-x₂ (при x₃=20)')
ax1.set_xlim(0, 100)
ax1.set_ylim(0, 50)
ax1.grid(True, alpha=0.3)
ax1.legend(loc='upper right', fontsize=8)

# Правая проекция: на плоскость x₁-x₃ (вид сбоку)
ax2 = axes[1]

x = np.linspace(0, 100, 200)
# При x₂=0 ограничения превращаются в:
z1 = (180 - x) / 3      # x₁ + 3x₃ = 180
z2 = (200 - 2*x) / 3    # 2x₁ + 3x₃ = 200

# Допустимая область
z_min = 20
z_max = np.minimum(z1, z2)
ax2.fill_between(x, z_min, z_max, where=(z_min <= z_max) & (x >= 0),
                 alpha=0.3, color='lightgray', label='Допустимая область')

ax2.plot(x, z1, 'b-', linewidth=2, label='x₁+3x₃=180')
ax2.plot(x, z2, 'g-', linewidth=2, label='2x₁+3x₃=200')
ax2.axhline(y=20, color='orange', linestyle='--', linewidth=2, label='x₃=20')

# Отмечаем проекции вершин (x₁, x₃)
for i, (coord, name, color) in enumerate(zip(V, V_NAMES, V_COLORS)):
    ax2.plot(coord[0], coord[2], 'o', color=color, markersize=10, markeredgecolor='black')
    ax2.annotate(name, (coord[0]+2, coord[2]), fontsize=8,
                 bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8))

ax2.set_xlabel('x₁ (столы)')
ax2.set_ylabel('x₃ (шкафы)')
ax2.set_title('Проекция на плоскость x₁-x₃ (при x₂=0)')
ax2.set_xlim(0, 100)
ax2.set_ylim(15, 50)
ax2.grid(True, alpha=0.3)
ax2.legend(loc='upper right', fontsize=8)

plt.suptitle('2D-проекции допустимой области', fontsize=14, y=1.02)
plt.tight_layout()

# -------------------------------------------------------------------
# 9. Инструкция по запуску
# -------------------------------------------------------------------

print('\n' + '='*60)
print('ИНСТРУКЦИЯ ПО ЗАПУСКУ')
print('='*60)
print('''

Управление анимацией:
  - Анимация запустится автоматически
  - Кадры сменяются каждые 2.5 секунды
  - Для повторного запуска закройте окно и откройте заново

Если анимация не работает:
  1. Проверьте версию matplotlib (нужна 3.3+)
  2. В Jupyter Notebook используйте %matplotlib notebook
  3. Для сохранения видео установите ffmpeg или pillow

Описание:
  Красный маркер с жёлтой заливкой — текущая вершина
  Фиолетовая линия — пройденный путь
  Полупрозрачные плоскости — ограничения задачи
  В правом верхнем углу — информация об итерации
''')

plt.show()