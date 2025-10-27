from simplex_solver import SimplexTwoPhase

def read_lp(path: str):
    with open(path, 'r', encoding='utf-8') as f:
        lines = [l.strip() for l in f if l.strip()]
    n = int(lines[0])
    typ, expr = lines[1].split(':', 1)
    constraints = lines[2:]
    return n, constraints, (typ.strip().lower(), expr.strip())

def main():
    n, cons, obj = read_lp('input.txt')
    solver = SimplexTwoPhase(n, cons, obj)

    sol = solver.solution()
    val = solver.objective_value()

    print("Оптимальное решение:")
    for k in sorted(sol):
        print(f"{k} = {sol[k]}")
    print(f"Оптимальное значение целевой функции: Z = {val}")

    with open('answer.txt', 'w', encoding='utf-8') as f:
        f.write("Оптимальное решение:\n")
        for k in sorted(sol):
            f.write(f"{k} = {sol[k]}\n")
        f.write(f"Z = {val}\n")

if __name__ == '__main__':
    main()
