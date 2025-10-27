from fractions import Fraction
from typing import List, Tuple

"""
ДВУХФАЗНЫЙ СИМПЛЕКС-МЕТОД.

Идея:
  • Phase I (вспомогательная задача): ищем ДОПУСТИМЫЙ БАЗИС.
    Для строк типа ">=" и "=" вводятся искусственные переменные r.
    Цель: min w = Σ r. Если оптимум w* > 0 → искусственные нельзя сделать нулевыми, система ограничений несовместна.
    Если w* = 0 → допустимый базис найден.

  • Удаляем столбцы искусственных переменных из таблицы (они не участвуют в основной задаче).

  • Phase II (основная задача): восстанавливаем исходную целевую функцию:
    ставим коэффициенты цели в нулевой строке, считаем дельты
    и крутим обычные симплекс-итерации до оптимальности
    (выбираем переменную по знаку дельты, делаем поворот для перехода в новый базис и повторяем, пока улучшений не будет).
"""

# --------- примитивы работы со строками таблицы ---------
def add_rows(r1, r2):
    """Покомпонентное сложение двух строк."""
    return [a + b for a, b in zip(r1, r2)]

def scale_row(c, r):
    """Умножение строки на константу (возвращает новую строку)."""
    return [c * a for a in r]

def argmax_pos(row):
    """
    Индекс максимальной положительной дельты (для задачи min).
    Последний элемент строки — это b (свободный член), его не рассматриваем.
    Если положительных нет → вернуть None (оптимум).
    """
    best, idx = Fraction(0), None
    for j in range(len(row) - 1):  # последний столбец — b
        if row[j] > best:
            best, idx = row[j], j
    return idx

def argmin_neg(row):
    """Индекс самой отрицательной дельты (для задачи max)."""
    best, idx = Fraction(0), None
    for j in range(len(row) - 1):
        if row[j] < best:
            best, idx = row[j], j
    return idx



# --------- парсинг выражений/ограничений ---------
def parse_linear_expr(expr: str, nvars: int) -> List[Fraction]:
    """
    Парсит '2x_1 - x_3 + x_2' в список коэффициентов длины nvars.
    Разбиение делаем по знакам, сохраняя их.
    Каждый токен вида "<коэфф> x_<индекс>". Пустой/плюсовой коэфф = 1, минус = -1.
    Индексация переменных в строке — с 1, а в массиве — с 0 ⇒ вычитаем единицу.
    """
    tokens = expr.replace('+', ' +').replace('-', ' -').split()
    coeffs = [Fraction(0) for _ in range(nvars)]
    for tok in tokens:
        if 'x_' not in tok:
            continue
        cpart, ipart = tok.split('x_')
        cpart = cpart.strip()
        if cpart in ('', '+'):
            c = Fraction(1)
        elif cpart == '-':
            c = Fraction(-1)
        else:
            c = Fraction(cpart)
        i = int(ipart) - 1
        if not (0 <= i < nvars):
            raise ValueError("Неверный индекс переменной")
        coeffs[i] += c
    return coeffs

def parse_constraint(line: str, nvars: int):
    """
    Парсит 'x_1 + 2x_2 <= 10' в (A, знак, b).
    Левая часть парсится как линейное выражение; правая — число (Fraction).
    """
    if '<=' in line:
        left, right = line.split('<='); sign = '<='
    elif '>=' in line:
        left, right = line.split('>='); sign = '>='
    elif '=' in line:
        left, right = line.split('=');  sign = '='
    else:
        raise ValueError("Ожидался <=, >= или =")
    a = parse_linear_expr(left.strip(), nvars)
    b = Fraction(right.strip())
    return a, sign, b

# --------- основной класс двухфазного симплекса ---------
class SimplexTwoPhase:
    """
    Табличная реализация двухфазного симплекса.
    На вход:
      num_vars - число исходных переменных (x_1..x_n),
      constraints - список строк-ограничений,
      objective - ('min'|'max', 'линейное выражение по x').

    Выход:
      методы solution() и objective_value().
    """

    def __init__(self, num_vars: int, constraints: List[str], objective: Tuple[str, str]):
        self.n = num_vars
        self.sense = objective[0].strip().lower()
        self.obj_expr = objective[1].strip()

        # Парсим систему ограничений в A (коэффициенты), знаки и b (правая часть).
        A, signs, b = [], [], []
        for line in constraints:
            a, s, rhs = parse_constraint(line, self.n)
            A.append(a); signs.append(s); b.append(rhs)

        # Нормируем строки: если b_i < 0 → домножаем на -1 и переворачиваем знак.
        # Это удобно, чтобы b был неотрицателен и проще было выбирать ведущую строку.
        for i in range(len(A)):
            if b[i] < 0:
                b[i] = -b[i]
                A[i] = [-c for c in A[i]]
                if signs[i] == '<=':
                    signs[i] = '>='
                elif signs[i] == '>=':
                    signs[i] = '<='

        # Считаем количество служебных столбцов: slack/surplus и искусственных.
        n_slack = sum(1 for s in signs if s in ('<=', '>='))
        n_art = sum(1 for s in signs if s in ('>=', '='))

        # Готовим пустую таблицу: [ x | slack | art | b ].
        self.slack_start = self.n
        self.art_start = self.n + n_slack
        ncols = self.n + n_slack + n_art + 1

        # таблица: (m+1) строк, где m = число ограничений
        self.T = [[Fraction(0) for _ in range(ncols)] for _ in range(len(A) + 1)]
        self.basic = [-1] * (len(A) + 1)  # базисный столбец на строку; basic[0] не используется
        # запомним индексы искусственных столбцов, чтобы потом удалить их
        self.artificial_cols = []

        # Заполняем строки ограничений; ставим базис: slack для <=, искусств. для >= и =.
        s_col = self.slack_start # следующий свободный столбец slack/surplus
        r_col = self.art_start

        for i, (a, s, rhs) in enumerate(zip(A, signs, b), start=1):
            # копируем коэффициенты исходных переменных x_j
            for j in range(self.n):
                self.T[i][j] = Fraction(a[j])
            self.T[i][-1] = Fraction(rhs) # свободный член

            # ставим служебные переменные и выбираем базис
            if s == '<=':
                self.T[i][s_col] = Fraction(1);
                self.basic[i] = s_col;
                s_col += 1

            # для "≥" ставим surplus (-1) и искусственную r (+1) → базис = r
            elif s == '>=':
                self.T[i][s_col] = Fraction(-1)
                self.T[i][r_col] = Fraction(1);
                self.basic[i] = r_col
                self.artificial_cols.append(r_col);
                s_col += 1;
                r_col += 1
            else:  # '='
                self.T[i][r_col] = Fraction(1);
                self.basic[i] = r_col
                self.artificial_cols.append(r_col);
                r_col += 1

        # ---------- Phase I: минимизируем сумму искусственных переменных ----------
        # Нулевая строка: ставим -1 в столбцах искусственных (эквивалент цели w).
        for col in self.artificial_cols:
            self.T[0][col] = Fraction(-1)

        # Приводим нулевую строку к каноническому виду:
        # для каждой строки, где r в базисе, просто прибавляем её к строке 0
        # (это обнулит коэффициенты при соответствующих r в строке Δ).
        for i in range(1, len(self.T)):
            if self.basic[i] in self.artificial_cols:
                self.T[0] = add_rows(self.T[0], self.T[i])

        # Симплекс-итерации Phase I (для min: пока есть Δ > 0).
        while True:
            enter = argmax_pos(self.T[0])
            if enter is None:
                break # все Δ <= 0 → достигли минимума w
            row = self._choose_row(enter) # строка выхода
            self._pivot(row, enter) # поворот: нормировка строки и зануление столбца

        # Если w != 0 — нет допустимых решений.
        if self.T[0][-1] != 0:
            raise ValueError("Система ограничений несовместна (Phase I: w > 0).")

        # Удаляем столбцы искусственных переменных (они больше не нужны).
        self._drop_artificial()

        # ---------- Phase II: восстанавливаем исходную цель и Δ ----------
        # Пусть исходная цель: min/max C^T x.
        # Формула редуцированных стоимостей: Δ = C_B * (базисные строки)-C.
        # Значение цели в RHS нулевой строки: C_B * (базисные b).
        obj = parse_linear_expr(self.obj_expr, self.n)

        # Начинаем со строки -C (ставим -коэффициенты цели в столбцы x_j).
        self.T[0] = [Fraction(0) for _ in range(len(self.T[0]))]
        for j, c in enumerate(obj):
            self.T[0][j] = Fraction(-c)

        # Прибавляем к строке 0 сумму c_b * (каждая базисная строка),
        # но только для тех строк, где базис - ИСХОДНАЯ переменная (не slack).
        for i in range(1, len(self.T)):
            bcol = self.basic[i]
            if 0 <= bcol < self.n:
                cb = Fraction(obj[bcol])
                if cb != 0:
                    self.T[0] = add_rows(self.T[0], scale_row(cb, self.T[i]))

        # Итерации до оптимальности: min — пока ∃ Δ>0; max — пока ∃ Δ<0.
        while True:
            enter = argmax_pos(self.T[0]) if self.sense == 'min' else argmin_neg(self.T[0])
            if enter is None:
                break # оптимум достигнут
            row = self._choose_row(enter)
            self._pivot(row, enter)

    # --------- служебные операции симплекса ---------
    def _choose_row(self, enter_col: int) -> int:
        """"
        Выбор ВЕДУЩЕЙ СТРОКИ. Используем правило: минимальный положительный Q = b_i / a_{i,enter}
        для всех строк с a_{i,enter} > 0. Если таких строк нет — целевая не ограничена.
        """
        best, row = None, None
        for i in range(1, len(self.T)):
            a = self.T[i][enter_col]
            if a > 0:
                val = self.T[i][-1] / a
                if best is None or val < best:
                    best, row = val, i
        if row is None:
            # Нет ни одной строки с a_{i,enter} > 0 → можно увеличивать x_enter бесконечно
            raise ValueError("Функция неограниченна (нет подходящей ведущей строки).")
        return row

    def _pivot(self, row: int, col: int):
        """Поворот вокруг опорного элемента: нормировка строки и зануление столбца."""
        piv = self.T[row][col]
        # нормировка ведущей строки
        self.T[row] = [v / piv for v in self.T[row]]
        # зануление столбца col в других строках (включая нулевую)
        for i in range(len(self.T)):
            if i == row:
                continue
            factor = self.T[i][col]
            if factor != 0:
                self.T[i] = [self.T[i][j] - factor * self.T[row][j] for j in range(len(self.T[i]))]
        # теперь в строке row базисная переменная — столбец col
        self.basic[row] = col

    def _drop_artificial(self):
        """Удаляет столбцы искусственных переменных и корректирует индексы базиса."""
        for col in sorted(self.artificial_cols, reverse=True):
            for i in range(len(self.T)):
                del self.T[i][col]
            for i in range(1, len(self.basic)):
                if self.basic[i] > col:
                    self.basic[i] -= 1

    # --------- результаты ---------
    def solution(self):
        """
        Возвращает значения x_i (не служебных) из текущего базиса.
        Читаем по столбцам: если x_j — единичный столбец и в базисе строки i → x_j = b_i, иначе 0
        """
        vals = [Fraction(0) for _ in range(self.n)]
        for i in range(1, len(self.T)):
            bcol = self.basic[i]
            if 0 <= bcol < self.n:
                vals[bcol] = self.T[i][-1]
        return {f"x_{i+1}": float(vals[i]) for i in range(self.n)}

    def objective_value(self) -> float:
        """Возвращает значение целевой функции (RHS верхней строки)."""
        return float(self.T[0][-1])
