def format2KorM(value):
    if value >= 1_000_000:
        return f'{value / 1_000_000:.0f}M'
    elif value >= 1_000:
        return f'{value / 1_000:.0f}K'
    else:
        return str(value)
    

def format2KorM_no100K(value):
    if value >= 1_000_000:
        return f'{value / 1_000_000:.1f}M' if value % 1_000_000 != 0 else f'{value // 1_000_000}M'
    elif value >= 100_000:  # 当 value ≥ 100K 时，转换为 M
        return f'{value / 1_000_000:.1f}M'
    elif value >= 1_000:
        return f'{value / 1_000:.0f}K'
    else:
        return str(value)
    

