import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np
import sympy as sp
from scipy.special import factorial
import time
import threading
from math import ceil
import cvxpy as cp
import os

# Стилизация интерфейса - цвета, шрифты и другие визуальные параметры
FONT_HEADER = ("Arial", 12, "bold")  # Шрифт для заголовков
FONT_SUBHEADER = ("Arial", 10, "bold")  # Шрифт для подзаголовков
FONT_NORMAL = ("Arial", 9)  # Основной шрифт
FONT_SMALL = ("Arial", 8)  # Мелкий шрифт
BG_COLOR = "#f0f0f0"  # Цвет фона
FG_COLOR = "#333333"  # Цвет текста
ENTRY_BG = "#ffffff"  # Цвет фона полей ввода
BUTTON_BG = "#4a7abc"  # Цвет фона кнопок
BUTTON_FG = "#ffffff"  # Цвет текста кнопок
FRAME_BG = "#e0e0e0"  # Цвет фона рамок

# Пути к иконкам
MAIN_ICON = "main.ico"
COEFF_ICON = "coeff_weights_ratios.ico"
PROGRESS_ICON = "progress.ico"

# Флаг для отслеживания первого открытия окна соотношений
first_ratios_open = True

# Сохраненные коэффициенты вместо полных функций
saved_coefficients = [
    [1.55, 0.8, 0.32, 0.1],
    [4.94, 8.2, 0.8, 0.6],
    [117, 345, 217, 19],
    [759, 1382, 917, 329],
    [0, 92, 13.8, 23],
    [13.57, 260, 0.33, 2.1],
    [83.23, 650, 26.87, 50.4],
    [65.74, 1.7, 6.1, 2],
    [1.8, 10.5, 6.3, 0.6]
]

# Веса по умолчанию для критериев и соотношений
saved_weights = [
    "0.31:0.9",    # weight_1
    "0.28:0.8",    # weight_2
    "0.1:0.78",    # weight_3
    "0.05:0.6",    # weight_4
    "0.04:0.4",    # weight_5
    "0.03:0.18",    # weight_6
    "0.03:0.17",    # weight_7
    "0.03:0.08",    # weight_8
    "0.01:0.07"     # weight_9
]

# Типы критериев по умолчанию
saved_types = ["1", "1", "1", "1", "1", "1", "1", "1", "1"]

# Сохраненные соотношения между критериями
saved_ratios = []   # saved_ratios = [(1, 3, 2, 1), (1, 4, 1, 1)]  # f2/f4 = 2/1, f2/f5 = 1/1

# Количество переменных по умолчанию
number_components = 4
number_characteristics = 9

# Сохраненные названия компонентов и характеристик
saved_component_names = [f"Комп-т {i + 1}" for i in range(10)]
saved_characteristic_names = [f"Хар-ка {i + 1}" for i in range(20)]

# Описания методов оптимизации для всплывающих подсказок
METHOD_DESCRIPTIONS = {
    "EDAS": "Метод оценки на основе расстояния от среднего решения. Эффективен когда важно отклонение от средних значений критериев.",
    "TOPSIS": "Метод упорядочения предпочтений по сходству с идеальным решением. Лучшая альтернатива должна быть ближе всего к идеалу.",
    "COPRAS": "Метод комплексной пропорциональной оценки. Одновременно учитывает положительные и отрицательные критерии.",
    "Quadratic": "Метод квадратичного программирования. Основан на математической оптимизации с квадратичной целевой функцией."
}

# Описания вариантов точности расчета
ITERATION_DESCRIPTIONS = {
    "Auto": "Автоматический подбор количества итераций. Рекомендуется для большинства задач.",
    "100": "Быстрый расчет с приблизительными результатами. Подходит для первоначального анализа.",
    "200": "Стандартная точность расчета. Хороший баланс между скоростью и точностью.",
    "400": "Высокая точность расчета. Рекомендуется для финальных вычислений.",
    "700": "Максимальная точность расчета. Может занять значительное время."
}


# Функция для установки иконки окна
def set_window_icon(window, icon_path):
    try:
        if os.path.exists(icon_path):
            window.iconbitmap(icon_path)
    except Exception as e:
        print(f"Не удалось загрузить иконку {icon_path}: {e}")

# Функция для показа всплывающей подсказки
def show_tooltip(event, text):
    # Создаем временное окно для подсказки
    tooltip = tk.Toplevel(main_window)
    tooltip.wm_overrideredirect(True)
    tooltip.wm_geometry(f"+{event.x_root + 10}+{event.y_root + 10}")

    # Настраиваем содержимое подсказки
    label = tk.Label(tooltip, text=text,
                     font=FONT_SMALL, bg="#ffffe0", relief="solid", borderwidth=1,
                     wraplength=300, justify=tk.LEFT)
    label.pack()

    # Функция для скрытия подсказки
    def hide_tooltip():
        tooltip.destroy()

    # Автоматически скрываем подсказку через 5 секунд
    tooltip.after(5000, hide_tooltip)

    # Также скрываем при движении мыши
    tooltip.bind("<Motion>", lambda e: hide_tooltip())


# Алгоритм оптимизации методом PSO (Particle Swarm Optimization) для максимизации
def pso_optimize_max(sympy_expr, variables, n_particles=50, n_iterations=100):
    # Преобразование символьного выражения в числовую функцию
    objective_func = sp.lambdify(variables, sympy_expr, 'numpy')
    n_vars = len(variables)

    # Обертка для обработки ошибок при вычислении
    def wrapped_func(x):
        try:
            result = objective_func(*x)
            # Обработка символьных выражений
            if isinstance(result, sp.Expr):
                if result.free_symbols:
                    return -np.inf  # Возвращаем бесконечность при наличии свободных символов
                return float(result.evalf())  # Вычисление численного значения
            return result
        except Exception as e:
            print(f"Error in wrapped_func: {e}")
            return -np.inf

    # Инициализация частиц - случайные позиции в пространстве решений
    particles = np.random.rand(n_particles, n_vars)
    particles = particles / np.sum(particles, axis=1, keepdims=True) * np.random.rand(n_particles, 1)
    velocities = np.random.rand(n_particles, n_vars) * 0.1  # Начальные скорости

    # Лучшие позиции частиц и глобально лучшая позиция
    personal_best_positions = particles.copy()
    personal_best_scores = np.array([wrapped_func(p) for p in particles])
    global_best_idx = np.argmax(personal_best_scores)
    global_best_position = personal_best_positions[global_best_idx]
    global_best_score = personal_best_scores[global_best_idx]

    # Параметры PSO
    w = 0.7  # Инерционный вес
    c1 = 1.5  # Когнитивный коэффициент
    c2 = 1.5  # Социальный коэффициент

    # Основной цикл оптимизации
    for _ in range(n_iterations):
        for i in range(n_particles):
            # Генерация случайных чисел для обновления скорости
            r1, r2 = np.random.rand(2)

            # Обновление скорости частицы
            velocities[i] = (w * velocities[i] +
                             c1 * r1 * (personal_best_positions[i] - particles[i]) +
                             c2 * r2 * (global_best_position - particles[i]))

            # Обновление позиции частицы с учетом границ [0, 1]
            particles[i] += velocities[i]
            particles[i] = np.clip(particles[i], 0, 1)

            # Нормировка суммы переменных
            total = np.sum(particles[i])
            if total > 1:
                particles[i] /= total

            # Оценка новой позиции
            current_score = wrapped_func(particles[i])

            # Обновление лучших позиций
            if current_score > personal_best_scores[i]:
                personal_best_positions[i] = particles[i].copy()
                personal_best_scores[i] = current_score
                if current_score > global_best_score:
                    global_best_score = current_score
                    global_best_position = particles[i].copy()

    return global_best_position, global_best_score


# Алгоритм PSO для минимизации (аналогичен предыдущему, но с изменением знаков)
def pso_optimize_min(sympy_expr, variables, n_particles=50, n_iterations=100):
    objective_func = sp.lambdify(variables, sympy_expr, 'numpy')
    n_vars = len(variables)

    def wrapped_func(x):
        try:
            result = objective_func(*x)
            if isinstance(result, sp.Expr):
                if result.free_symbols:
                    return np.inf  # Возвращаем +бесконечность при наличии свободных символов
                return float(result.evalf())
            return result
        except Exception as e:
            print(f"Error in wrapped_func: {e}")
            return np.inf

    # Инициализация частиц
    particles = np.random.rand(n_particles, n_vars)
    particles = particles / np.sum(particles, axis=1, keepdims=True) * np.random.rand(n_particles, 1)
    velocities = np.random.rand(n_particles, n_vars) * 0.1
    personal_best_positions = particles.copy()
    personal_best_scores = np.array([wrapped_func(p) for p in particles])
    global_best_idx = np.argmin(personal_best_scores)
    global_best_position = personal_best_positions[global_best_idx]
    global_best_score = personal_best_scores[global_best_idx]

    # Параметры PSO
    w = 0.7
    c1 = 1.5
    c2 = 1.5

    # Основной цикл оптимизации
    for _ in range(n_iterations):
        for i in range(n_particles):
            r1, r2 = np.random.rand(2)
            velocities[i] = (w * velocities[i] +
                             c1 * r1 * (personal_best_positions[i] - particles[i]) +
                             c2 * r2 * (global_best_position - particles[i]))

            particles[i] += velocities[i]
            particles[i] = np.clip(particles[i], 0, 1)

            total = np.sum(particles[i])
            if total > 1:
                particles[i] /= total

            current_score = wrapped_func(particles[i])
            if current_score < personal_best_scores[i]:
                personal_best_positions[i] = particles[i].copy()
                personal_best_scores[i] = current_score
                if current_score < global_best_score:
                    global_best_score = current_score
                    global_best_position = particles[i].copy()

    return global_best_position, global_best_score


# Генератор интервальных весов с дискретизацией
def generate_interval_weights_discrete(intervals, steps=2):
    # Проверяем наличие фиксированных значений
    fixed_values = []
    variable_indices = []
    for i, item in enumerate(intervals):
        if isinstance(item, (int, float)):
            fixed_values.append((i, float(item)))
        else:
            variable_indices.append(i)

    # Вычисляем сумму фиксированных значений
    fixed_sum = sum(val for _, val in fixed_values)

    # Генерируем оставшиеся веса
    remaining = 1.0 - fixed_sum
    if remaining < 0:
        return np.array([])  # Невозможно

    discrete_weights = []
    n_variable = len(variable_indices)

    # Если есть переменные веса
    if n_variable > 0:
        # Проверяем допустимость интервалов
        min_possible = sum(intervals[i][0] for i in variable_indices)
        max_possible = sum(intervals[i][1] for i in variable_indices)

        if min_possible > remaining or max_possible < remaining:
            return np.array([])  # Невозможно

        # Проверяем единственность решения
        unique_solution = None
        if abs(min_possible - remaining) < 1e-10:
            # Единственное решение - минимальные значения
            unique_weights = [0.0] * len(intervals)
            for idx, val in fixed_values:
                unique_weights[idx] = val
            for i in variable_indices:
                unique_weights[i] = intervals[i][0]
            unique_solution = unique_weights

        # Если найдено единственное решение
        if unique_solution is not None:
            return np.array([unique_solution])

        # Генерируем веса для переменных компонентов
        for _ in range(steps):
            weights = [0.0] * len(intervals)

            # Устанавливаем фиксированные значения
            for idx, val in fixed_values:
                weights[idx] = val

            # Генерируем переменные значения
            temp_weights = []
            total_temp = 0.0
            for i in variable_indices:
                low, high = intervals[i]
                w = np.random.uniform(low, high)
                temp_weights.append(w)
                total_temp += w

            # Нормализуем к оставшейся сумме
            if total_temp > 0:
                scale = remaining / total_temp
                for i, w in zip(variable_indices, temp_weights):
                    weights[i] = w * scale

            # Добавляем только уникальные комбинации
            if not any(np.allclose(weights, w) for w in discrete_weights):
                discrete_weights.append(weights)
    else:
        # Все веса фиксированные - одна точка
        weights = [val for _, val in fixed_values]
        if abs(sum(weights) - 1.0) < 1e-5:
            return np.array([weights])

    return np.array(discrete_weights)


# Реализация метода EDAS для непрерывных случаев
def interval_edas_for_functions(functions, weights, types, n_vars):
    # Количество независимых переменных
    n_independent = n_vars - 1

    # Фактор объема симплекса (n-1)!
    simplex_volume_factor = factorial(n_independent)

    # Создание символьных переменных
    x = sp.symbols(f'x1:{n_independent + 1}')

    processed_funcs = functions

    # Вычисление средних значений (AV) для каждого критерия
    AV = []
    for f in processed_funcs:
        # Подготовка пределов интегрирования
        integrals = []
        for i in range(n_independent):
            upper_limit = 1 - sum(x[j] for j in range(i))
            integrals.append((x[i], 0, upper_limit))

        # Многомерное интегрирование
        integral_result = f
        for i in range(n_independent - 1, -1, -1):
            var, lower, upper = integrals[i]
            integral_result = sp.integrate(integral_result, (var, lower, upper))

        # Корректировка объема симплекса
        integral_result *= simplex_volume_factor
        AV.append(integral_result)

    # Вычисление PDA (Positive Distance from Average) и NDA (Negative Distance from Average)
    PDA = []
    NDA = []
    for j, f in enumerate(processed_funcs):
        if types[j] == 1:  # Максимизация
            pda = (f - AV[j]) / AV[j]
            nda = (AV[j] - f) / AV[j]
        else:  # Минимизация
            pda = (AV[j] - f) / AV[j]
            nda = (f - AV[j]) / AV[j]
        PDA.append(pda)
        NDA.append(nda)

    # Вычисление SP (Sum of Positive Distances) и SN (Sum of Negative Distances)
    SP_expr = sum(w * p for w, p in zip(weights, PDA))
    SN_expr = sum(w * n for w, n in zip(weights, NDA))

    # Оптимизация SP и SN с помощью PSO
    best_pos_SP, max_SP = pso_optimize_max(SP_expr, x)
    best_pos_SN, max_SN = pso_optimize_max(SN_expr, x)

    # Нормировка SP и SN
    NSP = SP_expr / max_SP
    NSN = 1 - SN_expr / max_SN

    # Вычисление AS (Appraisal Score)
    AS = 0.5 * (NSP + NSN)

    # Оптимизация AS
    best_pos, _ = pso_optimize_max(AS, x)
    x_last_val = 1 - sum(best_pos)
    return np.append(best_pos, x_last_val)


# Реализация метода TOPSIS для непрерывных случаев
def interval_topsis_for_functions(functions, weights, types, n_vars):
    n_independent = n_vars - 1
    x = sp.symbols(f'x1:{n_independent + 1}')

    processed_funcs = functions

    # Нормализация функций
    R_exprs = []
    for f in processed_funcs:
        f_squared = f ** 2
        integral = f_squared
        vars_sorted = list(x)[::-1]  # Обратный порядок переменных

        # Последовательное интегрирование
        for i, var in enumerate(vars_sorted):
            if i < len(vars_sorted) - 1:
                upper_bound = 1 - sum(vars_sorted[i + 1:])
            else:
                upper_bound = 1
            integral = sp.integrate(integral, (var, 0, upper_bound))

        if integral == 0:
            R = f
        else:
            R = f / sp.sqrt(integral)
        R_exprs.append(R)

    # Взвешенные нормализованные функции
    V_exprs = [w * R for w, R in zip(weights, R_exprs)]

    # Поиск идеальных решений
    A_plus = []  # Положительное идеальное решение
    A_minus = []  # Отрицательное идеальное решение
    for j, V in enumerate(V_exprs):
        if types[j] == 1:  # Максимизация
            _, max_val = pso_optimize_max(V, x)
            _, min_val = pso_optimize_min(V, x)
            A_plus.append(max_val)
            A_minus.append(min_val)
        else:  # Минимизация
            _, min_val = pso_optimize_min(V, x)
            _, max_val = pso_optimize_max(V, x)
            A_plus.append(min_val)
            A_minus.append(max_val)

    # Вычисление расстояний до идеальных решений
    sum_plus = 0
    sum_minus = 0
    for j, V in enumerate(V_exprs):
        sum_plus += (V - A_plus[j]) ** 2
        sum_minus += (V - A_minus[j]) ** 2

    S_plus_expr = sp.sqrt(sum_plus)  # Расстояние до положительного идеального решения
    S_minus_expr = sp.sqrt(sum_minus)  # Расстояние до отрицательного идеального решения

    # Относительная близость к идеальному решению
    C_expr = S_minus_expr / (S_plus_expr + S_minus_expr)

    # Оптимизация относительной близости
    best_pos, _ = pso_optimize_max(C_expr, x)
    x_last_val = 1 - sum(best_pos)
    return np.append(best_pos, x_last_val)


# Реализация метода COPRAS для непрерывных случаев
def interval_copras_for_functions(functions, weights, types, n_vars):
    n_independent = n_vars - 1
    x = sp.symbols(f'x1:{n_independent + 1}')

    # Фактор объема симплекса
    simplex_volume_factor = factorial(n_independent)

    # Шаг 1: Нормализация критериев
    d_exprs = []
    for j, f in enumerate(functions):
        integral = f
        for i in range(n_independent - 1, -1, -1):
            var = x[i]
            upper_bound = 1 - sum(x[k] for k in range(i))
            integral = sp.integrate(integral, (var, 0, upper_bound))

        integral *= simplex_volume_factor

        if integral == 0:
            d_expr = weights[j] * f
        else:
            d_expr = weights[j] * f / integral
        d_exprs.append(d_expr)

    # Шаг 2: Разделение на максимизирующие и минимизирующие критерии
    S_plus_expr = 0  # Сумма для максимизирующих критериев
    S_minus_expr = 0  # Сумма для минимизирующих критериев
    for j in range(len(weights)):
        if types[j] == 1:  # Максимизация
            S_plus_expr += d_exprs[j]
        else:  # Минимизация
            S_minus_expr += d_exprs[j]

    # Шаг 3: Нахождение минимального значения S_minus
    _, min_S_minus = pso_optimize_min(S_minus_expr, x)

    # Шаг 4: Вычисление относительной важности минимизирующих критериев
    C_expr = min_S_minus / S_minus_expr if S_minus_expr != 0 else 0

    # Шаг 5: Расчет показателя альтернатив Q
    Q_expr = S_plus_expr + (min_S_minus * S_minus_expr) / (S_minus_expr * C_expr + 1e-10)

    # Шаг 6: Оптимизация Q
    best_pos, _ = pso_optimize_max(Q_expr, x)
    x_last_val = 1 - sum(best_pos)
    return np.append(best_pos, x_last_val)


# Реализация метода квадратичного программирования
def interval_quadratic_for_functions(functions, weights, types, n_vars):
    n_independent = n_vars - 1
    x_sym = sp.symbols(f'x1:{n_independent + 1}')
    x_last = 1 - sum(x_sym)  # Зависимая переменная

    # 1. Подготовка коэффициентов для целевой функции
    coeff_matrix = []
    for func in functions:
        coeffs = [0.0] * n_vars

        # Обработка линейных функций
        if func.is_Add or func.is_Mul:
            for i, var in enumerate(x_sym):
                coeffs[i] = float(func.coeff(var))
            coeffs[-1] = float(func.coeff(x_last))
        # Обработка квадратичных функций (соотношений)
        elif isinstance(func, sp.Pow) and func.exp == 2:
            base = func.base
            for i, var in enumerate(x_sym):
                coeffs[i] = float(base.coeff(var))
            coeffs[-1] = float(base.coeff(x_last))

        coeff_matrix.append(coeffs)

    coeff_matrix = np.array(coeff_matrix)
    n_criteria = coeff_matrix.shape[0]

    # 2. Нормализация коэффициентов
    normalized_matrix = np.zeros_like(coeff_matrix)
    for i in range(n_criteria):
        if normalization_type.get() == "sum":  # Нормализация по сумме
            norm_val = np.sum(np.abs(coeff_matrix[i]))
            if norm_val < 1e-10:
                norm_val = 1.0
            normalized_matrix[i] = coeff_matrix[i] / norm_val
        else:  # Нормализация по максимальному значению
            try:
                if types[i] == 1:
                    _, max_val = pso_optimize_max(functions[i], x_sym)
                else:
                    _, max_val = pso_optimize_max(functions[i], x_sym)
                if abs(max_val) < 1e-10:
                    normalized_matrix[i] = coeff_matrix[i]
                else:
                    normalized_matrix[i] = coeff_matrix[i] / abs(max_val)
            except:
                normalized_matrix[i] = coeff_matrix[i]

    # 3. Построение целевой функции для cvxpy
    x_cp = cp.Variable(n_vars)
    objective = 0

    for i in range(n_criteria):
        term = normalized_matrix[i] @ x_cp
        if types[i] == 1:  # Максимизация
            objective += weights[i] * term
        else:  # Минимизация
            objective += weights[i] * (1 - term)

    # 4. Ограничения задачи оптимизации
    constraints = [
        cp.sum(x_cp) == 1,  # Сумма переменных = 1
        x_cp >= 0,  # Неотрицательные значения
        x_cp <= 1  # Верхняя граница
    ]

    # 5. Решение задачи оптимизации
    problem = cp.Problem(cp.Maximize(objective), constraints)
    try:
        problem.solve(solver=cp.OSQP, verbose=False)  # Попытка решения первым решателем
    except:
        try:
            problem.solve(solver=cp.ECOS, verbose=False)  # Резервный решатель
        except:
            return np.ones(n_vars) / n_vars  # Равномерное распределение при ошибке

    # 6. Обработка результатов
    if x_cp.value is None:
        return np.ones(n_vars) / n_vars
    else:
        return x_cp.value


# Генерация символьной функции на основе коэффициентов
def generate_function_from_coefficients(coeffs, n_vars):
    n_independent = n_vars - 1
    x_sym = sp.symbols(f'x1:{n_independent + 1}')

    # Для метода Quadratic используем все переменные как независимые
    if method_var.get() == "Quadratic":
        expr = 0
        for i in range(n_vars):
            expr += coeffs[i] * sp.Symbol(f'x{i + 1}')
        return expr
    else:
        # Для других методов учитываем зависимую переменную
        x_last = 1 - sum(x_sym)
        expr = 0
        for i in range(n_independent):
            expr += coeffs[i] * x_sym[i]
        expr += coeffs[-1] * x_last
        return expr


# Парсинг строки соотношения в кортеж чисел
def parse_ratio(ratio_str):
    try:
        parts = list(map(float, ratio_str.split(":")))
        if len(parts) != 4:
            return None
        idx1, idx2, coef1, coef2 = parts
        if idx1 < 1 or idx2 < 1 or coef1 <= 0 or coef2 <= 0:
            return None
        return int(idx1) - 1, int(idx2) - 1, coef1, coef2
    except:
        return None


# Вычисление значений критериев для заданных вероятностей
def calculate_criteria_values(probabilities, functions, columns, saved_ratios, n_vars):
    # Создание символьных переменных
    symbols = sp.symbols(f'x1:{n_vars + 1}')

    # Словарь для подстановки значений
    sub_dict = {symbols[i]: float(probabilities[i]) for i in range(n_vars)}

    # Вычисление значений основных критериев
    f_values = []
    for i in range(columns):
        f_expr = functions[i]
        try:
            f_val = f_expr.subs(sub_dict)
            if f_val.free_symbols:
                f_val = f_val.evalf(subs=sub_dict)
            f_values.append(float(f_val))
        except:
            f_values.append(float('nan'))

    # Вычисление значений соотношений
    q_values = []
    for ratio in saved_ratios:
        i_idx, j_idx, coef1, coef2 = ratio
        try:
            if f_values[j_idx] == 0 or np.isnan(f_values[j_idx]) or np.isinf(f_values[j_idx]):
                ratio_val = float('inf')
            else:
                ratio_val = f_values[i_idx] / f_values[j_idx]
            target_val = coef1 / coef2
            # Расчет отклонения в процентах
            deviation = abs(ratio_val - target_val) / target_val * 100 if target_val != 0 else float('inf')
            q_values.append((ratio_val, target_val, deviation))
        except:
            q_values.append((float('nan'), coef1 / coef2, float('nan')))

    return f_values, q_values


# Обработчики интерфейса
def enter_functions(window):
    global saved_coefficients, include_ratios, saved_ratios, first_ratios_open, number_components
    global saved_component_names, saved_characteristic_names
    # Обновление количества компонентов из интерфейса
    try:
        number_components = int(spinbox_for_variables.get())
    except:
        number_components = 5

    # Сбор введенных коэффициентов
    saved_coefficients = []
    for i in range(len(coeff_vars)):
        coeffs = []
        for j in range(number_components):
            value = coeff_vars[i][j].get()
            try:
                num = float(value.replace(',', '.')) if value else 0.0
            except ValueError:
                messagebox.showwarning("Некорректный ввод",
                                       f"Некорректное значение в критерии {i + 1}, коэффициенте {j + 1}.\n"
                                       f"Используется значение 0.0")
                num = 0.0
            coeffs.append(num)
        saved_coefficients.append(coeffs)

    # Сохранение названий компонентов и характеристик
    saved_component_names = [v.get() for v in component_name_vars]
    saved_characteristic_names = [v.get() for v in characteristic_name_vars]

    window.grab_release()
    window.destroy()

    # Обновление флага включения соотношений
    include_ratios = include_ratios_var.get()

    # Получение количества соотношений
    try:
        num_ratios = int(number_ratios_var.get())
    except:
        num_ratios = 0

    num_ratios = max(0, min(10, num_ratios))

    # Отображение окна соотношений при необходимости
    if include_ratios and num_ratios > 0:
        show_ratios_window(num_ratios)
    else:
        weights_decision_making_matrices()


# Отображение окна для ввода соотношений
def show_ratios_window(num_ratios):
    global ratio_entries, ratios_window, first_ratios_open, saved_ratios

    # Использование сохраненных соотношений при первом открытии
    if first_ratios_open:
        saved_ratios = saved_ratios[:num_ratios]
        first_ratios_open = False

    ratios_window = tk.Toplevel()
    ratios_window.title(f"Соотношения между характеристиками")
    set_window_icon(ratios_window, COEFF_ICON)  # Установка иконки

    # Calculate dynamic size
    width = 380
    height = 150 + num_ratios * 40 + 50
    ratios_window.geometry(f"{width}x{height}")

    ratios_window.configure(bg=BG_COLOR)

    # Основной контейнер
    main_frame = tk.Frame(ratios_window, bg=BG_COLOR)
    main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

    # Заголовок
    title_frame = tk.Frame(main_frame, bg=FRAME_BG, height=40)
    title_frame.pack(fill=tk.X, padx=10, pady=(10, 0))
    tk.Label(title_frame, text=f"Введите соотношения между характеристиками",
             font=FONT_SUBHEADER, bg=FRAME_BG).pack(pady=5)

    # Создание области с прокруткой
    canvas = tk.Canvas(main_frame, bg=BG_COLOR)
    vsb = ttk.Scrollbar(main_frame, orient="vertical", command=canvas.yview)
    vsb.pack(side="right", fill="y")
    canvas.pack(side="left", fill="both", expand=True)
    canvas.configure(yscrollcommand=vsb.set)

    # Фрейм для содержимого
    content_frame = tk.Frame(canvas, bg=BG_COLOR)
    canvas.create_window((0, 0), window=content_frame, anchor="nw")

    # Обработчик изменения размера фрейма
    def on_frame_configure(event):
        canvas.configure(scrollregion=canvas.bbox("all"))

    content_frame.bind("<Configure>", on_frame_configure)

    # Инструкция по формату ввода
    tk.Label(content_frame, text="Формат: индекс1:индекс2:коэффициент1:коэффициент2",
             font=FONT_NORMAL, bg=BG_COLOR, fg=FG_COLOR).pack(anchor='w')
    tk.Label(content_frame, text="Пример: 2:4:3:1 означает хар-ка 2 к хар-ки 4 как 3/1",
             font=FONT_NORMAL, bg=BG_COLOR, fg=FG_COLOR).pack(anchor='w', pady=(0, 15))

    # Поля ввода для каждого соотношения
    ratio_entries = []
    for i in range(num_ratios):
        frame = tk.Frame(content_frame, bg=BG_COLOR)
        frame.pack(fill=tk.X, pady=5)
        tk.Label(frame, text=f"Соотношение {i + 1}:",
                 font=FONT_NORMAL, bg=BG_COLOR, fg=FG_COLOR).pack(side=tk.LEFT, padx=(0, 10))
        entry = tk.Entry(frame, width=20, font=FONT_NORMAL, bg=ENTRY_BG)

        # Заполнение предустановленными значениями
        if i < len(saved_ratios):
            idx1, idx2, coef1, coef2 = saved_ratios[i]
            entry.insert(0, f"{int(idx1) + 1}:{int(idx2) + 1}:{coef1}:{coef2}")

        entry.pack(side=tk.LEFT, padx=5)
        ratio_entries.append(entry)

    # Кнопка подтверждения
    btn_frame = tk.Frame(content_frame, bg=BG_COLOR)
    btn_frame.pack(pady=15)
    btn = tk.Button(btn_frame, text="Далее", font=FONT_NORMAL,
                    bg=BUTTON_BG, fg=BUTTON_FG, command=enter_ratios)
    btn.pack(padx=10, pady=5, ipadx=10, ipady=5)

    # Настройка прокрутки колесом мыши
    def on_mousewheel(event):
        if event.delta:
            canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

    canvas.bind_all("<MouseWheel>", on_mousewheel)

    ratios_window.grab_set()


# Обработка введенных соотношений
def enter_ratios():
    global saved_ratios
    saved_ratios = []
    for entry in ratio_entries:
        ratio_str = entry.get()
        if ratio_str.strip() == "":
            continue
        ratio = parse_ratio(ratio_str)
        if ratio:
            saved_ratios.append(ratio)
    ratios_window.grab_release()
    ratios_window.destroy()
    weights_decision_making_matrices()


# Сохранение весов и типов критериев
def bring_best_alternatives(window):
    global saved_weights, saved_types
    saved_weights = [w_var.get() for w_var in weight_vars]
    saved_types = [t_var.get() for t_var in type_vars]
    window.grab_release()
    window.destroy()
    run_optimization()


# Обновление информации о количестве критериев
def change_number_columns():
    value_label_for_columns.config(text="Выбрано критериев: " + spinbox_for_columns.get())


# Обновление информации о количестве компонентов
def change_number_components():
    pass  # Не требуется дополнительных действий


# Отображение окна для ввода коэффициентов критериев
def decision_making_matrices():
    global coeff_vars, include_ratios, component_name_vars, characteristic_name_vars
    include_ratios = include_ratios_var.get()

    columns = int(spinbox_for_columns.get())
    n_components = number_components  # Количество компонентов смеси
    n_independent = n_components - 1  # Количество независимых переменных

    window_DMM = tk.Toplevel()
    window_DMM.title("Коэффициенты смеси")
    set_window_icon(window_DMM, COEFF_ICON)  # Установка иконки

    # Calculate dynamic size
    width = 200 + (n_components + 1) * 80
    height = 100 + (columns + 2) * 45
    window_DMM.geometry(f"{width}x{height}")

    window_DMM.configure(bg=BG_COLOR)

    # Основной контейнер с возможностью прокрутки
    main_frame = tk.Frame(window_DMM, bg=BG_COLOR)
    main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

    # Заголовок
    title_frame = tk.Frame(main_frame, bg=FRAME_BG, height=40)
    title_frame.pack(fill=tk.X, pady=(0, 10))
    tk.Label(title_frame, text="Введите компоненты смеси и их характеристики",
             font=FONT_SUBHEADER, bg=FRAME_BG).pack(pady=8)

    # Создание области с прокруткой
    canvas = tk.Canvas(main_frame, bg=BG_COLOR)
    vsb = ttk.Scrollbar(main_frame, orient="vertical", command=canvas.yview)
    hsb = ttk.Scrollbar(main_frame, orient="horizontal", command=canvas.xview)
    vsb.pack(side="right", fill="y")
    hsb.pack(side="bottom", fill="x")
    canvas.pack(side="left", fill="both", expand=True)
    canvas.configure(yscrollcommand=vsb.set, xscrollcommand=hsb.set)

    # Фрейм для содержимого
    content_frame = tk.Frame(canvas, bg=BG_COLOR)
    canvas.create_window((0, 0), window=content_frame, anchor="nw")

    # Обработчик изменения размера фрейма
    def on_frame_configure(event):
        canvas.configure(scrollregion=canvas.bbox("all"))

    content_frame.bind("<Configure>", on_frame_configure)

    # Заголовки столбцов таблицы
    tk.Label(content_frame, text="Характеристики", font=FONT_NORMAL,
             bg=FRAME_BG, fg=FG_COLOR, width=15).grid(row=0, column=0, padx=5, pady=5, sticky='ew')

    # Поля ввода названий компонентов (столбцов)
    component_name_vars = []
    for i in range(n_components):
        default_name = saved_component_names[i] if i < len(saved_component_names) else f"Комп-т {i + 1}"
        var = tk.StringVar(value=default_name)
        component_name_vars.append(var)
        entry = tk.Entry(content_frame, textvariable=var, width=10,
                         font=FONT_NORMAL, bg=ENTRY_BG)
        entry.grid(row=0, column=i + 1, padx=5, pady=5, sticky='ew')

    # Поля ввода коэффициентов и названий характеристик (строк)
    coeff_vars = []
    characteristic_name_vars = []
    for i in range(columns):
        # Поле ввода названия характеристики
        default_name = saved_characteristic_names[i] if i < len(saved_characteristic_names) else f"Хар-ка {i + 1}"
        char_var = tk.StringVar(value=default_name)
        characteristic_name_vars.append(char_var)
        entry_char = tk.Entry(content_frame, textvariable=char_var, width=15,
                              font=FONT_NORMAL, bg=ENTRY_BG)
        entry_char.grid(row=i + 1, column=0, padx=5, pady=5, sticky='w')

        row_vars = []
        for j in range(n_independent):
            default_val = saved_coefficients[i][j] if i < len(saved_coefficients) and j < len(
                saved_coefficients[i]) else 0.0
            var = tk.StringVar(value=str(default_val))
            row_vars.append(var)
            entry = tk.Entry(content_frame, textvariable=var, width=10,
                             font=FONT_NORMAL, bg=ENTRY_BG)
            entry.grid(row=i + 1, column=j + 1, padx=5, pady=5)

        default_val = saved_coefficients[i][-1] if i < len(saved_coefficients) else 0.0
        var = tk.StringVar(value=str(default_val))
        row_vars.append(var)
        entry = tk.Entry(content_frame, textvariable=var, width=12,
                         font=FONT_NORMAL, bg=ENTRY_BG)
        entry.grid(row=i + 1, column=n_independent + 1, padx=5, pady=5)
        coeff_vars.append(row_vars)

    # Информационная подпись
    info_label = tk.Label(content_frame,
                          text="Введите коэффициенты рассматриваемой смеси",
                          font=FONT_SMALL, bg=BG_COLOR, fg="blue")
    info_label.grid(row=columns + 1, column=0, columnspan=n_independent + 2, pady=10)

    # Кнопка перехода к следующему шагу
    btn_frame = tk.Frame(content_frame, bg=BG_COLOR)
    btn_frame.grid(row=columns + 2, column=0, columnspan=n_independent + 2, pady=10)
    btn = tk.Button(btn_frame, text="Далее", font=FONT_NORMAL,
                    bg=BUTTON_BG, fg=BUTTON_FG, command=lambda: enter_functions(window_DMM))
    btn.pack(padx=10, pady=5, ipadx=15, ipady=5)

    # Настройка прокрутки колесом мыши
    def on_mousewheel(event):
        if event.delta:
            canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

    canvas.bind_all("<MouseWheel>", on_mousewheel)

    # Настройка размеров столбцов
    for col in range(1, n_independent + 2):
        content_frame.grid_columnconfigure(col, minsize=80)


# Отображение окна для ввода весов и типов критериев
def weights_decision_making_matrices():
    global weight_vars, type_vars
    columns = int(spinbox_for_columns.get())
    num_ratios = len(saved_ratios)
    total_columns = columns + num_ratios  # Основные + соотношения

    window_WDMM = tk.Toplevel()
    window_WDMM.title("Веса и типы характеристик смеси")
    set_window_icon(window_WDMM, COEFF_ICON)  # Установка иконки

    # Calculate dynamic size
    width = 600
    height = 150 + total_columns * 40 + 50
    window_WDMM.geometry(f"{width}x{height}")

    window_WDMM.configure(bg=BG_COLOR)

    weight_vars = []
    type_vars = []

    # Основной контейнер
    main_frame = tk.Frame(window_WDMM, bg=BG_COLOR)
    main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

    # Заголовок
    title_frame = tk.Frame(main_frame, bg=FRAME_BG, height=40)
    title_frame.pack(fill=tk.X, pady=(0, 10))
    tk.Label(title_frame, text=f"Задайте веса и типы для {total_columns} характеристик смеси",
             font=FONT_SUBHEADER, bg=FRAME_BG).pack(pady=8)

    # Создание области с прокруткой
    canvas = tk.Canvas(main_frame, bg=BG_COLOR)
    vsb = ttk.Scrollbar(main_frame, orient="vertical", command=canvas.yview)
    vsb.pack(side="right", fill="y")
    canvas.pack(side="left", fill="both", expand=True)
    canvas.configure(yscrollcommand=vsb.set)

    # Фрейм для содержимого
    content_frame = tk.Frame(canvas, bg=BG_COLOR)
    canvas.create_window((0, 0), window=content_frame, anchor="nw")

    # Обработчик изменения размера фрейма
    def on_frame_configure(event):
        canvas.configure(scrollregion=canvas.bbox("all"))

    content_frame.bind("<Configure>", on_frame_configure)

    # Функция проверки корректности введенных весов
    def validate_weights():
        for i, w_var in enumerate(weight_vars):
            weight_str = w_var.get()
            try:
                if ':' in weight_str:
                    # Проверка интервальных весов
                    parts = weight_str.split(':')
                    if len(parts) != 2:
                        return False, f"Неправильный формат весов для характеристики {i + 1}"
                    low, high = map(float, parts)
                    if low < 0 or high < 0:
                        return False, f"Веса должны быть положительными для характеристики {i + 1}"
                    if low > high:
                        return False, f"Минимальное значение больше максимального для характеристики {i + 1}"
                else:
                    # Проверка фиксированных весов
                    value = float(weight_str)
                    if value < 0:
                        return False, f"Вес должен быть положительным для характеристики {i + 1}"
            except ValueError:
                return False, f"Некорректное значение веса для характеристики {i + 1}"
        return True, ""

    # Поля ввода для основных критериев
    for i in range(columns):
        row_frame = tk.Frame(content_frame, bg=BG_COLOR)
        row_frame.pack(fill=tk.X, pady=5)

        char_name = saved_characteristic_names[i] if i < len(saved_characteristic_names) else f"Хар-ка {i + 1}"
        lbl_weight = tk.Label(row_frame, text=f"Важность {char_name} (min:max):",
                              font=FONT_NORMAL, bg=BG_COLOR, fg=FG_COLOR, width=25, anchor='w')
        lbl_weight.pack(side=tk.LEFT, padx=(10, 5))

        default_weight = saved_weights[i] if i < len(saved_weights) else "0:1"
        w_var = tk.StringVar(value=default_weight)
        weight_vars.append(w_var)
        entry_weight = tk.Entry(row_frame, textvariable=w_var, width=10,
                                font=FONT_NORMAL, bg=ENTRY_BG)
        entry_weight.pack(side=tk.LEFT, padx=5)

        lbl_type = tk.Label(row_frame, text="Тип (1=maximiz., -1=minimiz.):",
                            font=FONT_NORMAL, bg=BG_COLOR, fg=FG_COLOR, width=25, anchor='w')
        lbl_type.pack(side=tk.LEFT, padx=(20, 5))

        default_type = saved_types[i] if i < len(saved_types) else "1"
        t_var = tk.StringVar(value=default_type)
        type_vars.append(t_var)
        entry_type = tk.Entry(row_frame, textvariable=t_var, width=5,
                              font=FONT_NORMAL, bg=ENTRY_BG)
        entry_type.pack(side=tk.LEFT, padx=5)

    # Поля ввода для соотношений (дополнительные критерии)
    for i, ratio in enumerate(saved_ratios):
        idx = columns + i
        idx1, idx2, coef1, coef2 = ratio
        char1 = saved_characteristic_names[idx1] if idx1 < len(saved_characteristic_names) else f"f{idx1 + 1}"
        char2 = saved_characteristic_names[idx2] if idx2 < len(saved_characteristic_names) else f"f{idx2 + 1}"

        row_frame = tk.Frame(content_frame, bg=BG_COLOR)
        row_frame.pack(fill=tk.X, pady=5)

        lbl_weight = tk.Label(row_frame,
                              text=f"Важность соот-ния {char1}/{char2}={coef1}:{coef2} (min:max):",
                              font=FONT_NORMAL, bg=BG_COLOR, fg="blue", width=25, anchor='w')
        lbl_weight.pack(side=tk.LEFT, padx=(10, 5))

        default_weight = saved_weights[idx] if idx < len(saved_weights) else "0.1:0.1"
        w_var = tk.StringVar(value=default_weight)
        weight_vars.append(w_var)
        entry_weight = tk.Entry(row_frame, textvariable=w_var, width=10,
                                font=FONT_NORMAL, bg=ENTRY_BG)
        entry_weight.pack(side=tk.LEFT, padx=5)

        lbl_type = tk.Label(row_frame, text="Тип: minimiz. (фиксировано)",
                            font=FONT_NORMAL, bg=BG_COLOR, fg="blue", width=25, anchor='w')
        lbl_type.pack(side=tk.LEFT, padx=(20, 5))

        t_var = tk.StringVar(value="-1")
        type_vars.append(t_var)
        entry_type = tk.Entry(row_frame, textvariable=t_var, width=5,
                              font=FONT_NORMAL, bg=ENTRY_BG, state='disabled')
        entry_type.pack(side=tk.LEFT, padx=5)

    # Кнопка запуска оптимизации
    btn_frame = tk.Frame(content_frame, bg=BG_COLOR)
    btn_frame.pack(fill=tk.X, pady=20)

    def bring_best_alternatives():
        # Проверка корректности весов
        valid, error_msg = validate_weights()
        if not valid:
            messagebox.showerror("Ошибка", error_msg)
            return

        # Проверка суммы минимальных и максимальных значений
        min_sum = 0.0
        max_sum = 0.0

        for w_var in weight_vars:
            weight_str = w_var.get()
            if ':' in weight_str:
                low, high = map(float, weight_str.split(':'))
                min_sum += low
                max_sum += high
            else:
                value = float(weight_str)
                min_sum += value
                max_sum += value

        if min_sum > 1.0 + 1e-5:
            messagebox.showerror("Ошибка",
                                 f"Сумма минимальных весов ({min_sum:.4f}) превышает 1.0.\n"
                                 "Уменьшите минимальные значения весов.")
            return

        if max_sum < 1.0 - 1e-5:
            messagebox.showerror("Ошибка",
                                 f"Сумма максимальных весов ({max_sum:.4f}) меньше 1.0.\n"
                                 "Увеличьте максимальные значения весов.")
            return

        # Сохранение весов и типов
        global saved_weights, saved_types
        saved_weights = [w_var.get() for w_var in weight_vars]
        saved_types = [t_var.get() for t_var in type_vars]

        window_WDMM.grab_release()
        window_WDMM.destroy()
        run_optimization()

    btn = tk.Button(btn_frame, text="Оптимизировать", font=FONT_NORMAL,
                    bg=BUTTON_BG, fg=BUTTON_FG, command=bring_best_alternatives)
    btn.pack(pady=10, ipadx=20, ipady=7)

    # Настройка прокрутки колесом мыши
    def on_mousewheel(event):
        if event.delta:
            canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

    canvas.bind_all("<MouseWheel>", on_mousewheel)


# Запуск процесса оптимизации
def run_optimization():
    # Замер общего времени выполнения
    start_total_time = time.time()

    columns = int(spinbox_for_columns.get())
    method = method_var.get()
    n_vars = number_components  # Используем количество компонентов как количество переменных
    total_columns = columns + len(saved_ratios)  # Основные + соотношения

    # Генерация символьных функций для критериев
    functions = []
    for i in range(columns):
        if i < len(saved_coefficients):
            expr = generate_function_from_coefficients(saved_coefficients[i], n_vars)
            functions.append(expr)
        else:
            functions.append(sp.sympify("0"))

    # Добавление функций для соотношений (квадраты невязок)
    for ratio in saved_ratios:
        i, j, a, b = ratio
        if i < len(functions) and j < len(functions):
            deviation = b * functions[i] - a * functions[j]
            functions.append(deviation ** 2)  # Квадрат невязки как дополнительный критерий

    # Парсинг весов и типов критериев
    intervals = []
    types = []
    for i in range(total_columns):
        try:
            s = saved_weights[i]
            if ':' in s:
                parts = s.split(':')
                if len(parts) == 2:
                    low, high = map(float, parts)
                    intervals.append((low, high))
                else:
                    # Пробуем интерпретировать как фиксированное значение
                    try:
                        value = float(s)
                        intervals.append(value)
                    except:
                        intervals.append((0.0, 0.5))  # Значение по умолчанию
            else:
                # Фиксированное значение
                try:
                    value = float(s)
                    intervals.append(value)
                except:
                    intervals.append((0.0, 0.5))  # Значение по умолчанию
            types.append(int(saved_types[i]))
        except:
            messagebox.showerror("Ошибка", "Неправильный формат весов или типов")
            return

    # Создание окна прогресса
    progress_window = tk.Toplevel(main_window)
    progress_window.title("Расчет оптимальных решений")
    progress_window.geometry('480x310')
    set_window_icon(progress_window, PROGRESS_ICON)  # Установка иконки
    progress_window.configure(bg=BG_COLOR)
    progress_window.grab_set()

    # Заголовок окна прогресса
    title_frame = tk.Frame(progress_window, bg=FRAME_BG, height=40)
    title_frame.pack(fill=tk.X)
    tk.Label(title_frame, text="Выполнение оптимизации",
             font=FONT_SUBHEADER, bg=FRAME_BG).pack(pady=8)

    # Основное содержимое окна прогресса
    content_frame = tk.Frame(progress_window, bg=BG_COLOR)
    content_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=15)

    # Анимация прогресса
    canvas = tk.Canvas(content_frame, width=400, height=30, bg=BG_COLOR, highlightthickness=0)
    canvas.pack(pady=5)

    progress_var = tk.DoubleVar()
    progress_bar = ttk.Progressbar(content_frame, variable=progress_var, length=400)
    progress_bar.pack(pady=10)

    # Информационные метки
    lbl_iterations = tk.Label(content_frame,
                              text="Выполнено: 0 итераций",
                              font=FONT_NORMAL, bg=BG_COLOR, fg=FG_COLOR)
    lbl_iterations.pack(pady=5)

    lbl_stability = tk.Label(content_frame,
                             text="Стабильность: 0/0",
                             font=FONT_NORMAL, bg=BG_COLOR, fg=FG_COLOR)
    lbl_stability.pack(pady=5)

    lbl_status = tk.Label(content_frame,
                          text="Подготовка к расчету...",
                          font=FONT_NORMAL, bg=BG_COLOR, fg=FG_COLOR)
    lbl_status.pack(pady=5)

    # Флаг и функция для отмены расчета
    cancel_flag = [False]

    def cancel_calculation():
        cancel_flag[0] = True
        btn_cancel.config(state=tk.DISABLED)
        lbl_status.config(text="Отмена...")

    # Кнопка отмены
    btn_frame = tk.Frame(content_frame, bg=BG_COLOR)
    btn_frame.pack(pady=10)
    btn_cancel = tk.Button(btn_frame, text="Отменить", font=FONT_NORMAL,
                           bg="#e74c3c", fg=BUTTON_FG, command=cancel_calculation)
    btn_cancel.pack(pady=5, ipadx=15, ipady=5)

    animation_id = [None]

    # Функция анимации прогресса
    def update_animation(frame):
        canvas.delete("all")
        canvas.create_rectangle(0, 0, 400, 30, outline="#ddd", fill="#f0f0f0")
        pos = frame % 400
        canvas.create_rectangle(pos - 30, 5, pos + 30, 25, fill="#4caf50", outline="#4caf50")

        dots = "." * ((frame // 10) % 4)
        lbl_status.config(text=f"Выполняются расчеты{dots}")

        if not cancel_flag[0]:
            animation_id[0] = canvas.after(100, update_animation, frame + 5)

    progress_window.update()
    update_animation(0)

    # Переменные для хранения результатов
    best_alternatives = []
    solutions = []  # Сохранение векторов решений
    start_time = time.time()
    stability_threshold = 5  # Количество итераций без изменений для стабилизации
    top_k = 3  # Количество лучших альтернатив для сравнения
    stability_count = 0
    previous_top_k = None

    # Основная функция расчета в отдельном потоке
    def run_calculation():
        nonlocal stability_count, previous_top_k
        btn_cancel.config(state=tk.NORMAL)
        steps_val = number_iterations.get()

        # Замер времени начала расчета
        start_calculation_time = time.time()

        try:
            # Режим автоматического подбора (Auto)
            if steps_val == "Auto":
                # Проверяем, есть ли единственное решение для весов?
                test_weights = generate_interval_weights_discrete(intervals, steps=2)
                if len(test_weights) == 0:
                    messagebox.showerror("Ошибка", "Не удалось сгенерировать веса")
                    return

                # Проверяем, единственное ли решение
                if len(test_weights) == 1:
                    # Генерируем веса (единственный набор)
                    weights = test_weights[0]
                    try:
                        if method == "EDAS":
                            scores = interval_edas_for_functions(functions, weights, types, n_vars)
                        elif method == "TOPSIS":
                            scores = interval_topsis_for_functions(functions, weights, types, n_vars)
                        elif method == "COPRAS":
                            scores = interval_copras_for_functions(functions, weights, types, n_vars)
                        elif method == "Quadratic":
                            scores = interval_quadratic_for_functions(functions, weights, types, n_vars)

                        best_alt = np.argmax(scores) + 1
                        best_alternatives.append(best_alt)
                        solutions.append(scores)

                        # Показываем результат сразу, без цикла стабилизации
                        progress_var.set(1)
                        lbl_iterations.config(text=f"Выполнено: 1 итерация (единственное решение)")
                        lbl_stability.config(text="Стабильность: 1/1")
                        lbl_status.config(text="Найдено единственное решение весов")
                        progress_window.update()
                        time.sleep(1)  # Даем время увидеть сообщение
                    except Exception as e:
                        messagebox.showerror("Ошибка", f"Ошибка оптимизации: {str(e)}")
                        return
                else:
                    # Итеративный процесс с проверкой стабильности
                    stability_threshold_val = 5
                    total_done = 0
                    progress_bar.configure(maximum=stability_threshold_val)
                    progress_var.set(0)

                    # Хранилище уникальных весов (для избежания повторений)
                    generated_weights = set()
                    max_attempts = 100  # Максимальное количество попыток генерации уникальных весов

                    # Основной цикл автоматического режима
                    while stability_count < stability_threshold_val and not cancel_flag[0]:
                        attempts = 0
                        weights = None

                        # Генерация нового уникального набора весов
                        while attempts < max_attempts:
                            # Генерируем ОДИН набор весов
                            weights_list = generate_interval_weights_discrete(intervals, steps=1)

                            if len(weights_list) == 0:
                                continue  # Пропускаем итерацию, если не удалось сгенерировать веса

                            candidate = weights_list[0]  # Берем первый (и единственный) набор

                            # Создаем уникальный хэш для весов (с округлением для избежания проблем с float)
                            weights_hash = tuple(round(w, 6) for w in candidate)

                            # Проверяем, были ли такие веса уже сгенерированы
                            if weights_hash not in generated_weights:
                                weights = candidate
                                generated_weights.add(weights_hash)
                                break

                            attempts += 1

                        # Если не удалось сгенерировать уникальные веса
                        if weights is None:
                            lbl_status.config(text="Не удалось сгенерировать уникальные веса")
                            progress_window.update()
                            break

                        if cancel_flag[0]:
                            break

                        try:
                            if method == "EDAS":
                                scores = interval_edas_for_functions(functions, weights, types, n_vars)
                            elif method == "TOPSIS":
                                scores = interval_topsis_for_functions(functions, weights, types, n_vars)
                            elif method == "COPRAS":
                                scores = interval_copras_for_functions(functions, weights, types, n_vars)
                            elif method == "Quadratic":
                                scores = interval_quadratic_for_functions(functions, weights, types, n_vars)

                            best_alt = np.argmax(scores) + 1
                            best_alternatives.append(best_alt)
                            solutions.append(scores)
                            total_done += 1
                        except Exception:
                            continue

                        # Проверка стабильности результатов
                        if best_alternatives:
                            unique, counts = np.unique(best_alternatives, return_counts=True)
                            sorted_indices = np.argsort(-counts)
                            current_top_k = unique[sorted_indices][:min(top_k, len(unique))]

                            if previous_top_k is not None and np.array_equal(previous_top_k, current_top_k):
                                stability_count += 1
                            else:
                                stability_count = 0

                            previous_top_k = current_top_k

                        # Обновление интерфейса прогресса
                        progress_var.set(stability_count)
                        lbl_iterations.config(text=f"Выполнено: {total_done} итераций")
                        lbl_stability.config(text=f"Стабильность: {stability_count}/{stability_threshold_val}")

                        progress_window.update()
                        if stability_count >= stability_threshold_val:
                            break
                        time.sleep(0.1)  # Небольшая задержка для обновления интерфейса

            # Режим с фиксированным числом итераций
            else:
                steps_val = int(steps_val)
                # Генерируем все наборы весов сразу
                weights_list = generate_interval_weights_discrete(intervals, steps=steps_val)
                total_weights = len(weights_list)
                if total_weights == 0:
                    messagebox.showerror("Ошибка", "Не удалось сгенерировать веса")
                    return

                progress_bar.configure(maximum=total_weights)
                progress_var.set(0)

                # Основной цикл для фиксированного числа итераций
                for i, weights in enumerate(weights_list):
                    if cancel_flag[0]:
                        break

                    try:
                        if method == "EDAS":
                            scores = interval_edas_for_functions(functions, weights, types, n_vars)
                        elif method == "TOPSIS":
                            scores = interval_topsis_for_functions(functions, weights, types, n_vars)
                        elif method == "COPRAS":
                            scores = interval_copras_for_functions(functions, weights, types, n_vars)
                        elif method == "Quadratic":
                            scores = interval_quadratic_for_functions(functions, weights, types, n_vars)

                        best_alt = np.argmax(scores) + 1
                        best_alternatives.append(best_alt)
                        solutions.append(scores)

                        # Обновление интерфейса прогресса
                        progress_var.set(i + 1)
                        lbl_iterations.config(text=f"Выполнено: {i + 1} / {total_weights} итераций")

                        # Расчет оставшегося времени
                        elapsed = time.time() - start_calculation_time
                        if i > 0:
                            avg_time = elapsed / i
                            remaining = avg_time * (total_weights - i - 1)
                            mins, secs = divmod(ceil(remaining), 60)
                            time_str = f"{mins} мин {secs} сек" if mins > 0 else f"{secs} сек"
                            lbl_stability.config(text=f"Примерное время: {time_str}")

                    except Exception:
                        continue

                    progress_window.update()

            # Завершение анимации
            if animation_id[0]:
                canvas.after_cancel(animation_id[0])

            progress_window.grab_release()
            progress_window.destroy()

            # Рассчет общего времени выполнения
            total_time = time.time() - start_calculation_time

            # Форматирование времени в удобном виде
            if total_time < 60:
                time_str = f"{total_time:.2f} секунд"
            elif total_time < 3600:
                minutes = int(total_time // 60)
                seconds = total_time % 60
                time_str = f"{minutes} минут {seconds:.1f} секунд"
            else:
                hours = int(total_time // 3600)
                minutes = int((total_time % 3600) // 60)
                seconds = total_time % 60
                time_str = f"{hours} часов {minutes} минут {seconds:.1f} секунд"

            # Обработка результатов после завершения расчета
            if not cancel_flag[0] and best_alternatives:
                # Анализ результатов
                unique, counts = np.unique(best_alternatives, return_counts=True)
                total = len(best_alternatives)
                probabilities = counts / total

                sorted_indices = np.argsort(-probabilities)
                unique_sorted = unique[sorted_indices]
                probabilities_sorted = probabilities[sorted_indices]

                # Определение ширины столбцов
                first_column_width = 20  # Ширина для "Компонент смеси"
                second_column_width = 10  # Ширина для "Ее доля"

                # Формирование заголовка
                header1 = "Компонент смеси"
                header2 = "Ее доля"
                header = header1.ljust(first_column_width) + " | " + header2.rjust(second_column_width - 1)

                # Формирование разделителя
                separator = "-" * (first_column_width + 5) + "--+--" + "-" * second_column_width

                # Формирование строк данных
                table_rows = []
                for alt, prob in zip(unique_sorted, probabilities_sorted):
                    comp_name = saved_component_names[alt - 1] if (alt - 1) < len(saved_component_names) else str(alt)
                    alt_str = comp_name.ljust(first_column_width + 9)
                    prob_str = f"{prob:.6f}".rjust(second_column_width)
                    row = alt_str + " | " + prob_str
                    table_rows.append(row)

                # Сборка таблицы
                formatted_table = "\n".join([header, separator] + table_rows)

                # Формирование строки результатов
                result_str = f"Метод: {method}\n"
                result_str += f"Всего итераций: {total}\n"
                result_str += f"Время выполнения: {time_str}\n\n"
                result_str += "Решение:\n"
                result_str += formatted_table + "\n\n"

                # Вычисление значений критериев для лучшей альтернативы
                f_values, q_values = calculate_criteria_values(
                    solutions[0] if solutions else np.zeros(n_vars),
                    functions,
                    columns,
                    saved_ratios,
                    n_vars
                )

                result_str += f"Значения характеристик смеси:\n"
                for i, f_val in enumerate(f_values[:columns]):
                    char_name = saved_characteristic_names[i] if i < len(saved_characteristic_names) else f"f{i + 1}"
                    result_str += f"  {char_name} = {f_val:.6f}\n"

                # Добавление информации о соотношениях
                if saved_ratios:
                    result_str += "\nСоотношения в получившейся смеси:\n"
                    for j, ratio in enumerate(saved_ratios):
                        i_idx, j_idx, coef1, coef2 = ratio
                        char1 = saved_characteristic_names[i_idx] if i_idx < len(
                            saved_characteristic_names) else f"f{i_idx + 1}"
                        char2 = saved_characteristic_names[j_idx] if j_idx < len(
                            saved_characteristic_names) else f"f{j_idx + 1}"
                        ratio_val, target_val, deviation = q_values[j]
                        result_str += f"  q{j + 1}: {char1}/{char2} = {ratio_val:.6f} "
                        result_str += f"(целевое: {coef1}/{coef2}, "
                        result_str += f"отклонение: {deviation:.2f}%)\n"

                # Отображение результатов
                messagebox.showinfo("Результат", result_str)
            elif cancel_flag[0]:
                messagebox.showinfo("Отменено", "Расчет прерван пользователем")
            else:
                messagebox.showinfo("Ошибка", "Не удалось получить результаты")

        except Exception as e:
            messagebox.showerror("Ошибка", f"Произошла ошибка при расчетах: {str(e)}")

    # Запуск потока расчета
    calculation_thread = threading.Thread(target=run_calculation)
    calculation_thread.daemon = True
    calculation_thread.start()


# Функция для показа справки
def show_help():
    help_window = tk.Toplevel(main_window)
    help_window.title("Справка по программе")
    help_window.geometry("650x700")
    set_window_icon(help_window, MAIN_ICON)  # Установка иконки
    help_window.configure(bg=BG_COLOR)
    help_window.resizable(True, True)

    # Главный контейнер
    main_frame = tk.Frame(help_window, bg=BG_COLOR)
    main_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)

    # Заголовок (центрируем)
    title_frame = tk.Frame(main_frame, bg=BG_COLOR)
    title_frame.pack(fill=tk.X, pady=(0, 15))

    title_label = tk.Label(title_frame, text="Инструкция по использованию программы",
                           font=FONT_HEADER, bg=BG_COLOR, fg=FG_COLOR)
    title_label.pack()

    # Создание текстового поля с прокруткой
    text_frame = tk.Frame(main_frame, bg=BG_COLOR)
    text_frame.pack(fill=tk.BOTH, expand=True)

    # Текстовое поле с прокруткой
    text_widget = tk.Text(text_frame, wrap=tk.WORD, font=FONT_NORMAL,
                          bg=BG_COLOR, fg=FG_COLOR, padx=15, pady=15,
                          relief=tk.FLAT, highlightthickness=1,
                          highlightcolor="#cccccc", highlightbackground="#cccccc")

    # Настройка прокрутки
    vsb = ttk.Scrollbar(text_frame, orient="vertical", command=text_widget.yview)
    text_widget.configure(yscrollcommand=vsb.set)

    # Упаковка элементов
    text_widget.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
    vsb.pack(side=tk.RIGHT, fill=tk.Y)

    # Вставка текста инструкции
    instruction_text = """
    1. О программе
    Эта программа предназначена для многокритериальной оптимизации состава смесей. 
    Она позволяет найти оптимальное соотношение компонентов смеси, учитывая несколько 
    критериев (характеристик) и возможные соотношения между ними.

    2. Методы оптимизации
    Программа включает четыре метода многокритериальной оптимизации:

    2.1. EDAS (Evaluation based on Distance from Average Solution)
    Метод оценки на основе расстояния от среднего решения. Этот подход:
    - Сначала вычисляет среднее решение по всем критериям
    - Затем измеряет положительное и отрицательное расстояние каждой альтернативы от среднего значения
    - Рассчитывает оценку на основе комбинации этих расстояний
    - Особенно эффективен, когда важно отклонение от средних значений критериев
    - Хорошо работает, когда трудно определить идеальные и наихудшие значения
    - Устойчив к выбросам в данных, так как использует средние значения

    2.2. TOPSIS (Technique for Order of Preference by Similarity to Ideal Solution)
    Метод упорядочения предпочтений по сходству с идеальным решением. Этот метод:
    - Определяет идеальное решение (лучшие значения по всем критериям)
    - Определяет наихудшее решение (худшие значения по всем критериям)
    - Вычисляет расстояние каждой альтернативы до идеального и наихудшего решений
    - Ранжирует альтернативы по их относительной близости к идеальному решению
    - Особенно эффективен, когда можно четко определить "идеальную" и "наихудшую" альтернативы
    - Интуитивно понятен, так как лучшая альтернатива должна быть ближе всего к идеалу

    2.3. COPRAS (Complex Proportional Assessment)
    Метод комплексной пропорциональной оценки. Этот подход:
    - Одновременно учитывает как положительные (максимизируемые), так и отрицательные (минимизируемые) критерии
    - Вычисляет относительную важность каждой альтернативы на основе ее вклада в общую эффективность
    - Использует нормализацию для приведения различных критериев к сопоставимому масштабу
    - Особенно полезен, когда необходимо сбалансировать противоречивые критерии
    - Позволяет оценить как прямую, так и относительную эффективность альтернатив
    - Хорошо подходит для задач, где важны пропорциональные соотношения между критериями

    2.4. Quadratic (Квадратичное программирование)
    Метод квадратичного программирования. Этот подход:
    - Основан на математической оптимизации с квадратичной целевой функцией и линейными ограничениями
    - Использует специализированные алгоритмы (OSQP, ECOS) для эффективного решения задач оптимизации
    - Обеспечивает высокую точность для задач, где зависимости между переменными можно аппроксимировать квадратичными функциями
    - Эффективно работает с большим количеством переменных и ограничений
    - Менее интуитивно понятен для пользователей без математического образования
    - Требует меньшие вычислительных ресурсов

    3. Заполнение параметров

    3.1. Основные параметры задачи
    - Количество компонентов смеси: укажите число составляющих смеси (например, 
      количество ингредиентов в рецепте или химических компонентов).
    - Количество характеристик смеси: укажите число критериев оценки (например, 
      вкус, стоимость, питательная ценность и т.д.).
    - Учитывать соотношения характеристик: отметьте, если нужно учитывать 
      определенные соотношения между характеристиками.

    3.2. Ввод компонентов и характеристик
    На следующем экране вам нужно будет заполнить таблицу коэффициентов:
    - Каждая строка соответствует одной характеристике смеси
    - Каждый столбец соответствует доле компонента в смеси

    Пример: для смеси из 3 компонентов (A, B, C) с характеристикой "вкус":
    - Коэффициент для A: 0.5
    - Коэффициент для B: 0.3
    - Коэффициент для C: 0.6

    3.3. Соотношения между характеристиками
    Если включена опция учета соотношений, можно задать отношения между парами 
    характеристик в формате: индекс1:индекс2:коэффициент1:коэффициент2

    Пример: "2:4:3:1" означает, что характеристика 2 должна относиться к 
    характеристике 4 как 3:1.

    3.4. Веса и типы характеристик
    Для каждой характеристики нужно задать:
    - Важность (вес) в формате min:max (например, "0.2:0.8") или конкретное значение
    - Тип критерия: 1 для максимизации, -1 для минимизации

    4. Настройки расчета
    - Точность расчета: выбор между автоматическим подбором и фиксированным 
      количеством итераций. Для большинства задач рекомендуется "Auto".
    - Тип нормализации: доступен только для метода Quadratic.

    5. Запуск расчета
    После нажатия кнопки "Оптимизировать" программа выполнит расчет и покажет 
    результаты в виде вероятностей оптимального состава смеси.

    6. Интерпретация результатов
    Результаты представляются в виде таблицы, где для каждого компонента смеси 
    указывается вероятность его оптимальной доли. Также выводятся значения 
    всех характеристик для полученной смеси и отклонения от заданных соотношений.

    Советы:
    - Начинайте с небольшого числа характеристик и компонентов
    - Для первых тестов используйте метод TOPSIS или EDAS
    - Внимательно проверяйте введенные коэффициенты на корректность
    """

    text_widget.insert(tk.END, instruction_text.strip())
    text_widget.config(state=tk.DISABLED)  # Делаем текст только для чтения

    # Кнопка закрытия (центрируем)
    button_frame = tk.Frame(main_frame, bg=BG_COLOR)
    button_frame.pack(pady=15)

    btn_close = tk.Button(button_frame, text="Закрыть", font=FONT_NORMAL,
                          bg=BUTTON_BG, fg=BUTTON_FG, command=help_window.destroy,
                          width=15, height=1)
    btn_close.pack()

    # Установка фокуса на текстовое поле и прокрутка в начало
    text_widget.see("1.0")

    # Настройка минимального размера окна
    help_window.minsize(600, 500)


# Основное окно приложения
main_window = tk.Tk()
main_window.title("Система поддержки принятия решений для оптимизации многокомпонентных смесей")
main_window.geometry("700x630")
set_window_icon(main_window, MAIN_ICON)  # Установка иконки для главного окна
main_window.configure(bg=BG_COLOR)

# Инициализация переменных интерфейса
method_var = tk.StringVar(value="EDAS")  # Метод оптимизации по умолчанию
number_iterations = tk.StringVar(value="Auto")  # Режим расчета по умолчанию
include_ratios_var = tk.BooleanVar(value=False)  # Флаг включения соотношений
normalization_type = tk.StringVar(value="max")  # Тип нормализации для Quadratic
number_ratios_var = tk.StringVar(value="1")  # Количество соотношений по умолчанию

# Главный контейнер
main_frame = tk.Frame(main_window, bg=BG_COLOR)
main_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)

# Заголовок приложения
header_frame = tk.Frame(main_frame, bg=FRAME_BG, height=50)
header_frame.pack(fill=tk.X, pady=(0, 15))
tk.Label(header_frame, text="Многокритериальная оптимизация состава смесей",
         font=FONT_HEADER, bg=FRAME_BG).pack(pady=12)

# Раздел 1: Основные параметры задачи
section1 = tk.LabelFrame(main_frame, text="1. Основные параметры задачи",
                         font=FONT_SUBHEADER, bg=BG_COLOR, fg=FG_COLOR)
section1.pack(fill=tk.X, pady=10, ipadx=10, ipady=10)

# Количество компонентов смеси
var_frame = tk.Frame(section1, bg=BG_COLOR)
var_frame.pack(fill=tk.X, padx=10, pady=5)
tk.Label(var_frame, text="Количество компонентов смеси:",
         font=FONT_NORMAL, bg=BG_COLOR, fg=FG_COLOR).pack(side=tk.LEFT, padx=(0, 10))
spinbox_for_variables = ttk.Spinbox(var_frame, from_=2, to=10, width=5, font=FONT_NORMAL)
spinbox_for_variables.pack(side=tk.LEFT)
spinbox_for_variables.set(str(number_components))
tk.Label(var_frame, text="(состав смеси (замляника, крыжовник, малина, ...))",
         font=FONT_SMALL, bg=BG_COLOR, fg="#666666").pack(side=tk.LEFT, padx=(10, 0))

# Количество характеристик смеси
crit_frame = tk.Frame(section1, bg=BG_COLOR)
crit_frame.pack(fill=tk.X, padx=10, pady=5)
tk.Label(crit_frame, text="Количество характеристик смеси:",
         font=FONT_NORMAL, bg=BG_COLOR, fg=FG_COLOR).pack(side=tk.LEFT, padx=(0, 10))
spinbox_for_columns = ttk.Spinbox(crit_frame, from_=1, to=20, width=5, font=FONT_NORMAL)
spinbox_for_columns.pack(side=tk.LEFT)
spinbox_for_columns.set(str(number_characteristics))
value_label_for_columns = tk.Label(crit_frame, text="(элементы смеси, показатели, ...)",
                                   font=FONT_SMALL, bg=BG_COLOR, fg="#666666")
value_label_for_columns.pack(side=tk.LEFT, padx=(10, 0))

# Флаг учета соотношений характеристик
ratio_frame = tk.Frame(section1, bg=BG_COLOR)
ratio_frame.pack(fill=tk.X, padx=10, pady=5)
tk.Label(ratio_frame, text="Учитывать соотношения характеристик:",
         font=FONT_NORMAL, bg=BG_COLOR, fg=FG_COLOR).pack(side=tk.LEFT, padx=(0, 10))
chk_ratios = tk.Checkbutton(ratio_frame, variable=include_ratios_var,
                            bg=BG_COLOR, activebackground=BG_COLOR)
chk_ratios.pack(side=tk.LEFT)

# Количество соотношений
ratio_count_frame = tk.Frame(section1, bg=BG_COLOR)
tk.Label(ratio_count_frame, text="Количество соотношений:",
         font=FONT_NORMAL, bg=BG_COLOR, fg=FG_COLOR).pack(side=tk.LEFT, padx=(0, 10))
spinbox_ratios = ttk.Spinbox(ratio_count_frame, from_=1, to=9, width=5,
                             textvariable=number_ratios_var, font=FONT_NORMAL)
spinbox_ratios.pack(side=tk.LEFT)
tk.Label(ratio_count_frame, text="(1-9)",
         font=FONT_SMALL, bg=BG_COLOR, fg="#666666").pack(side=tk.LEFT, padx=(10, 0))

# Раздел 2: Выбор метода оптимизации
section2 = tk.LabelFrame(main_frame, text="2. Выбор метода оптимизации",
                         font=FONT_SUBHEADER, bg=BG_COLOR, fg=FG_COLOR)
section2.pack(fill=tk.X, pady=10, ipadx=10, ipady=10)

method_frame = tk.Frame(section2, bg=BG_COLOR)
method_frame.pack(fill=tk.X, padx=10, pady=5)
tk.Label(method_frame, text="Метод оптимизации:",
         font=FONT_NORMAL, bg=BG_COLOR, fg=FG_COLOR).pack(side=tk.LEFT, padx=(0, 10))

# Доступные методы оптимизации
methods = [
    ("EDAS - Оценка на основе среднего решения", "EDAS"),
    ("TOPSIS - Техника приближения к идеальному решению", "TOPSIS"),
    ("COPRAS - Комплексная оценка пропорциональных альтернатив", "COPRAS"),
    ("Quadratic - Квадратичное программирование", "Quadratic")
]

method_desc = tk.Label(method_frame, text="", wraplength=500, justify=tk.LEFT,
                       font=FONT_NORMAL, bg=BG_COLOR, fg="#444444")
method_desc.pack(side=tk.LEFT, padx=(10, 0))


# Функция обновления описания метода
def update_method_desc(event):
    method = method_var.get()
    for name, value in methods:
        if value == method:
            method_desc.config(text=name.split(" - ")[1])
            break


# Выпадающий список методов
method_combo = ttk.Combobox(method_frame, textvariable=method_var,
                            values=[m[1] for m in methods], state="readonly",
                            width=12, font=FONT_NORMAL)
method_combo.pack(side=tk.LEFT)
method_combo.bind("<<ComboboxSelected>>", update_method_desc)
method_combo.bind("<Enter>", lambda e: show_tooltip(e, METHOD_DESCRIPTIONS[method_var.get()]))
method_combo.bind("<Leave>", lambda e: main_window.focus())
method_combo.current(0)
update_method_desc(None)

# Раздел 3: Настройки расчета
section3 = tk.LabelFrame(main_frame, text="3. Настройки расчета",
                         font=FONT_SUBHEADER, bg=BG_COLOR, fg=FG_COLOR)
section3.pack(fill=tk.X, pady=10, ipadx=10, ipady=10)

# Настройка точности расчета
iter_frame = tk.Frame(section3, bg=BG_COLOR)
iter_frame.pack(fill=tk.X, padx=10, pady=5)
tk.Label(iter_frame, text="Точность расчета:",
         font=FONT_NORMAL, bg=BG_COLOR, fg=FG_COLOR).pack(side=tk.LEFT, padx=(0, 10))

# Варианты точности расчета
iterations = [
    ("Auto - Автоматический подбор (рекомендуется)", "Auto"),
    ("100 - Быстрый расчет (приблизительный)", "100"),
    ("200 - Стандартная точность", "200"),
    ("400 - Высокая точность", "400"),
    ("700 - Максимальная точность (долгий расчет)", "700")
]

iter_desc = tk.Label(iter_frame, text="", wraplength=400, justify=tk.LEFT,
                     font=FONT_NORMAL, bg=BG_COLOR, fg="#444444")
iter_desc.pack(side=tk.LEFT, padx=(10, 0))


# Функция обновления описания точности
def update_iter_desc(event):
    iter_val = number_iterations.get()
    for name, value in iterations:
        if value == iter_val:
            iter_desc.config(text=name.split(" - ")[1])
            break


# Выпадающий список вариантов точности
spinbox_iter = ttk.Combobox(iter_frame, textvariable=number_iterations,
                            values=[i[1] for i in iterations], width=8,
                            state="readonly", font=FONT_NORMAL)
spinbox_iter.pack(side=tk.LEFT)
spinbox_iter.bind("<<ComboboxSelected>>", update_iter_desc)
spinbox_iter.bind("<Enter>", lambda e: show_tooltip(e, ITERATION_DESCRIPTIONS[number_iterations.get()]))
spinbox_iter.bind("<Leave>", lambda e: main_window.focus())
spinbox_iter.current(0)
update_iter_desc(None)

# Настройки нормализации для метода Quadratic
normalization_frame = tk.Frame(section3, bg=BG_COLOR)
normalization_frame.pack(fill=tk.X, padx=10, pady=5)
tk.Label(normalization_frame, text="Тип нормализации:",
         font=FONT_NORMAL, bg=BG_COLOR, fg=FG_COLOR).pack(side=tk.LEFT, padx=(0, 10))
rb_sum = tk.Radiobutton(normalization_frame, text="По сумме", variable=normalization_type,
                        value="sum", font=FONT_NORMAL, bg=BG_COLOR, fg=FG_COLOR)
rb_sum.pack(side=tk.LEFT, padx=(0, 10))
rb_max = tk.Radiobutton(normalization_frame, text="По максимуму", variable=normalization_type,
                        value="max", font=FONT_NORMAL, bg=BG_COLOR, fg=FG_COLOR)
rb_max.pack(side=tk.LEFT)

# Раздел 4: Управление
section4 = tk.LabelFrame(main_frame, text="4. Запуск расчета",
                         font=FONT_SUBHEADER, bg=BG_COLOR, fg=FG_COLOR)
section4.pack(fill=tk.X, pady=15, ipadx=10, ipady=10)

# Информационная метка
info_label = tk.Label(section4,
                      text="После ввода коэффициентов можно задать соотношения между характеристиками",
                      font=FONT_SMALL, bg=BG_COLOR, fg="blue", wraplength=500)
info_label.pack(pady=5)

# Фрейм для кнопок
button_frame = tk.Frame(section4, bg=BG_COLOR)
button_frame.pack(pady=10)

# Кнопка справки
btn_help = tk.Button(button_frame, text="?", font=("Arial", 12, "bold"),
                     bg="#ff9800", fg="white", width=3, command=show_help)
btn_help.pack(side=tk.LEFT, padx=5)

# Кнопка запуска ввода коэффициентов
btn_matrix = tk.Button(button_frame, text="Ввести компоненты и характеристики",
                       font=FONT_NORMAL, bg=BUTTON_BG, fg=BUTTON_FG,
                       command=decision_making_matrices)
btn_matrix.pack(side=tk.LEFT, padx=5)


# Функции управления видимостью элементов интерфейса
def update_normalization_visibility(*args):
    """Обновляет видимость настроек нормализации в зависимости от выбранного метода"""
    if method_var.get() == "Quadratic":
        normalization_frame.pack(fill=tk.X, padx=10, pady=5)
    else:
        normalization_frame.pack_forget()


def update_info_label_visibility(*args):
    """Обновляет видимость информационной метки в зависимости от флага соотношений"""
    if include_ratios_var.get():
        info_label.pack(pady=5)
    else:
        info_label.pack_forget()


def toggle_ratio_count_visibility(*args):
    """Переключает видимость блока количества соотношений"""
    if include_ratios_var.get():
        ratio_count_frame.pack(fill=tk.X, padx=10, pady=5)
    else:
        ratio_count_frame.pack_forget()


# Привязка обработчиков изменений
method_var.trace_add("write", lambda *args: update_normalization_visibility())
include_ratios_var.trace_add("write", lambda *args: update_info_label_visibility())
include_ratios_var.trace_add("write", lambda *args: toggle_ratio_count_visibility())


# Обработчики изменений для спинбоксов
def change_number_columns(event=None):
    value_label_for_columns.config(text="Выбрано критериев: " + spinbox_for_columns.get())


def change_number_components(event=None):
    pass  # Не требуется дополнительных действий


# Инициализация видимости элементов
update_normalization_visibility()
update_info_label_visibility()
toggle_ratio_count_visibility()

# Привязка обработчиков к элементам управления
spinbox_for_columns.bind("<<Increment>>", lambda e: change_number_columns())
spinbox_for_columns.bind("<<Decrement>>", lambda e: change_number_columns())
spinbox_for_columns.bind("<KeyRelease>", lambda e: change_number_columns())

spinbox_for_variables.bind("<<Increment>>", lambda e: change_number_components())
spinbox_for_variables.bind("<<Decrement>>", lambda e: change_number_components())
spinbox_for_variables.bind("<KeyRelease>", lambda e: change_number_components())

# Запуск главного цикла приложения
main_window.update()
main_window.minsize(700, 630)
main_window.mainloop()
