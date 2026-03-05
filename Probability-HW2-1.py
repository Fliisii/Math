"""
Задача Бюффона об игле: численная оценка числа π

Метод: 
1. Генерируем случайное положение и ориентацию иглы
2. Проверяем, пересекает ли игла хотя бы одну линию
3. Оцениваем π по формуле π ≈ 2n/m, где n - число бросков, m - число пересечений
4. Строим график сходимости оценки
5. Оцениваем доверительный интервал методом бутстрепа
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import time

# -------------------------------------------------------------------
# 1. Параметры эксперимента
# -------------------------------------------------------------------

# Количество бросков (10^9 = 1 миллиард)
N_TOTAL = 10**9

# Размер блока для построения графика (10^6 = 1 миллион)
BLOCK_SIZE = 10**6

# Отношение длины иглы к расстоянию между линиями
# Классический случай: l = d
l_over_d = 1.0

print("="*60)
print("ЗАДАЧА БЮФФОНА ОБ ИГЛЕ")
print("="*60)
print(f"\nПараметры эксперимента:")
print(f"  Общее число бросков: {N_TOTAL:,}")
print(f"  Размер блока: {BLOCK_SIZE:,}")
print(f"  l/d = {l_over_d}")

# -------------------------------------------------------------------
# 2. Проведение эксперимента (исправленная версия)
# -------------------------------------------------------------------

def run_buffon_experiment(n_trials, block_size):
    """
    Проводит эксперимент Бюффона.
    Исправлена ошибка с пересечением линий.
    """
    
    # Линии на расстоянии 1 друг от друга: ..., -1, 0, 1, 2, ...
    # Игла длины l = 1 (так как l/d = 1)
    
    cumulative_m = []      # накопленные пересечения
    cumulative_pi = []      # накопленные оценки π
    total_m = 0             # счётчик пересечений
    
    num_blocks = n_trials // block_size
    
    print(f"\nВыполняется эксперимент из {num_blocks} блоков по {block_size:,} бросков...")
    start_time = time.time()
    
    for block in range(num_blocks):
        # x0 - расстояние от центра иглы до ближайшей линии СЛЕВА (0 ≤ x0 < 1)
        x0 = np.random.uniform(0, 1, block_size)
        
        # θ - угол наклона иглы (0 ≤ θ < π)
        theta = np.random.uniform(0, np.pi, block_size)
        
        # Длина проекции иглы на перпендикуляр к линиям
        half_length = 0.5
        projection = half_length * np.abs(np.sin(theta))
        
        # ИСПРАВЛЕНИЕ: игла пересекает линию, если:
        # 1) расстояние до левой линии ≤ проекции ИЛИ
        # 2) расстояние до правой линии (1 - x0) ≤ проекции
        crosses = (x0 <= projection) | ((1 - x0) <= projection)
        
        # Обновляем счётчики
        block_m = np.sum(crosses)
        total_m += block_m
        
        # Сохраняем накопленные значения
        cumulative_m.append(total_m)
        
        # Оценка π на данный момент
        if total_m > 0:
            # Формула: π = 2n/m (при l = d)
            pi_est = 2 * (block+1) * block_size / total_m
            cumulative_pi.append(pi_est)
        else:
            cumulative_pi.append(np.nan)
        
        # Прогресс
        if (block + 1) % 10 == 0:
            elapsed = time.time() - start_time
            print(f"  Блок {block+1}/{num_blocks}, пересечений: {total_m}, π ≈ {cumulative_pi[-1]:.6f}, время: {elapsed:.1f}с")
    
    total_time = time.time() - start_time
    print(f"\nЭксперимент завершён за {total_time:.1f} секунд")
    
    return np.array(cumulative_m), np.array(cumulative_pi), total_m

# Запускаем эксперимент
cumulative_m, cumulative_pi, total_m = run_buffon_experiment(N_TOTAL, BLOCK_SIZE)

# Итоговая оценка π
pi_estimate = 2 * N_TOTAL / total_m

print(f"\nРЕЗУЛЬТАТЫ:")
print(f"  Всего бросков: {N_TOTAL:,}")
print(f"  Пересечений: {total_m:,}")
print(f"  Частота пересечений: {total_m/N_TOTAL:.6f}")
print(f"  Оценка π: {pi_estimate:.8f}")
print(f"  Истинное π: {np.pi:.8f}")
print(f"  Погрешность: {abs(pi_estimate - np.pi):.8f} ({abs(pi_estimate - np.pi)/np.pi*100:.4f}%)")

# -------------------------------------------------------------------
# 3. Построение графика сходимости
# -------------------------------------------------------------------

# Для графика берём не все точки (иначе будет 1000 точек, что многовато)
# Выбираем примерно 15 точек, равномерно распределённых в логарифмическом масштабе
n_points = 15
indices = np.unique(np.logspace(0, np.log10(len(cumulative_pi)-1), n_points, dtype=int))

plot_trials = indices * BLOCK_SIZE
plot_pi = cumulative_pi[indices]

plt.figure(figsize=(12, 8))

# Основной график
plt.subplot(2, 1, 1)
plt.semilogx(plot_trials, plot_pi, 'bo-', linewidth=2, markersize=8, label='Оценка π')
plt.axhline(y=np.pi, color='r', linestyle='--', linewidth=2, label=f'Истинное π = {np.pi:.8f}')
plt.xlabel('Число бросков', fontsize=12)
plt.ylabel('Оценка числа π', fontsize=12)
plt.title('Сходимость оценки числа π в задаче Бюффона', fontsize=14)
plt.grid(True, alpha=0.3, which='both')
plt.legend(fontsize=11)

# Добавляем значения на график
for i, (trial, pi_val) in enumerate(zip(plot_trials, plot_pi)):
    plt.annotate(f'{pi_val:.4f}', (trial, pi_val), 
                xytext=(5, 5), textcoords='offset points', fontsize=8, alpha=0.7)

# График погрешности
plt.subplot(2, 1, 2)
error = np.abs(plot_pi - np.pi)
plt.semilogx(plot_trials, error, 'go-', linewidth=2, markersize=6)
plt.semilogx(plot_trials, 1/np.sqrt(plot_trials), 'r--', linewidth=2, 
             label='Теоретическая ошибка ~ 1/√n')
plt.xlabel('Число бросков', fontsize=12)
plt.ylabel('Абсолютная погрешность', fontsize=12)
plt.title('Погрешность оценки', fontsize=14)
plt.grid(True, alpha=0.3, which='both')
plt.legend(fontsize=11)

plt.tight_layout()
plt.savefig('buffon_convergence.png', dpi=150, bbox_inches='tight')
plt.show()

# -------------------------------------------------------------------
# 4. Бутстреп для доверительного интервала
# -------------------------------------------------------------------

print("\n" + "="*60)
print("БУТСТРЕП-ОЦЕНКА ДОВЕРИТЕЛЬНОГО ИНТЕРВАЛА")
print("="*60)

# Для бутстрепа нам нужны исходные данные, но хранить 1e9 значений невозможно.
# Вместо этого мы используем тот факт, что в каждом блоке у нас было block_m пересечений.
# Мы можем сгенерировать булевы значения для каждого блока, аппроксимируя распределение.

# Восстанавливаем количество пересечений по блокам
block_crosses = np.diff(cumulative_m, prepend=0)
blocks = len(block_crosses)

print(f"\nАнализ {blocks} блоков по {BLOCK_SIZE:,} бросков")
print(f"Среднее число пересечений в блоке: {np.mean(block_crosses):.2f}")
print(f"Стандартное отклонение: {np.std(block_crosses):.2f}")

# Бутстреп: многократно генерируем выборки блоков с повторением
n_bootstrap = 10000
bootstrap_pi = []

print(f"\nВыполняется бутстреп ({n_bootstrap} итераций)...")
start_bs = time.time()

for i in range(n_bootstrap):
    # Выбираем блоки с повторением
    sample_indices = np.random.choice(blocks, blocks, replace=True)
    sample_crosses = block_crosses[sample_indices]
    total_crosses_bs = np.sum(sample_crosses)
    
    # Оценка π
    if total_crosses_bs > 0:
        pi_bs = 2 * blocks * BLOCK_SIZE / total_crosses_bs
        bootstrap_pi.append(pi_bs)
    
    if (i + 1) % 1000 == 0:
        print(f"  Итерация {i+1}/{n_bootstrap}")

bootstrap_pi = np.array(bootstrap_pi)

# 95% доверительный интервал (перцентильный метод)
ci_lower = np.percentile(bootstrap_pi, 2.5)
ci_upper = np.percentile(bootstrap_pi, 97.5)

print(f"\nРЕЗУЛЬТАТЫ БУТСТРЕПА:")
print(f"  Оценка π (основная): {pi_estimate:.8f}")
print(f"  95% доверительный интервал: [{ci_lower:.8f}, {ci_upper:.8f}]")
print(f"  Ширина интервала: {ci_upper - ci_lower:.8f}")
print(f"  Истинное π входит в интервал? {ci_lower <= np.pi <= ci_upper}")

# Гистограмма бутстреп-оценок
plt.figure(figsize=(10, 6))
plt.hist(bootstrap_pi, bins=50, density=True, alpha=0.7, color='skyblue', edgecolor='black')
plt.axvline(x=np.pi, color='red', linestyle='--', linewidth=2, label=f'Истинное π = {np.pi:.6f}')
plt.axvline(x=pi_estimate, color='blue', linestyle='-', linewidth=2, label=f'Оценка = {pi_estimate:.6f}')
plt.axvline(x=ci_lower, color='green', linestyle=':', linewidth=2, label=f'95% ДИ: [{ci_lower:.6f}, {ci_upper:.6f}]')
plt.axvline(x=ci_upper, color='green', linestyle=':', linewidth=2)

plt.xlabel('Оценка числа π', fontsize=12)
plt.ylabel('Плотность вероятности', fontsize=12)
plt.title('Бутстреп-распределение оценки числа π', fontsize=14)
plt.grid(True, alpha=0.3)
plt.legend(fontsize=10)

plt.tight_layout()
plt.savefig('buffon_bootstrap.png', dpi=150, bbox_inches='tight')
plt.show()

# -------------------------------------------------------------------
# 5. Сравнение с теоретическим распределением
# -------------------------------------------------------------------

print("\n" + "="*60)
print("АНАЛИЗ РЕЗУЛЬТАТОВ")
print("="*60)

# Теоретическая вероятность пересечения
p_theoretical = 2 / np.pi
p_observed = total_m / N_TOTAL

print(f"\nТеоретическая вероятность пересечения: {p_theoretical:.8f}")
print(f"Наблюдаемая частота: {p_observed:.8f}")
print(f"Относительная погрешность: {abs(p_observed - p_theoretical)/p_theoretical*100:.4f}%")

# Проверка нормальности распределения числа пересечений в блоках
from scipy import stats

# Статистика Колмогорова-Смирнова
ks_statistic, ks_pvalue = stats.kstest(block_crosses, 'norm', 
                                        args=(np.mean(block_crosses), np.std(block_crosses)))

print(f"\nПроверка нормальности распределения числа пересечений в блоках:")
print(f"  Среднее: {np.mean(block_crosses):.2f}")
print(f"  Дисперсия: {np.var(block_crosses):.2f}")
print(f"  Теоретическая дисперсия (np(1-p)): {BLOCK_SIZE * p_theoretical * (1 - p_theoretical):.2f}")
print(f"  KS-статистика: {ks_statistic:.4f}")
print(f"  p-value: {ks_pvalue:.4f}")
print(f"  Распределение {'нормально' if ks_pvalue > 0.05 else 'не нормально'} на уровне 0.05")

print("""


Примечания:
  - Для 10^9 бросков требуется ~2-3 минуты на современном компьютере
  - При недостатке оперативной памяти можно уменьшить N_TOTAL
  - Результаты сохраняются в файлы buffon_convergence.png и buffon_bootstrap.png
""")