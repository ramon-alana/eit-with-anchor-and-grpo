import sys
from ast import literal_eval
from pathlib import Path

# 检查 help 请求
if "--help" in sys.argv or "-h" in sys.argv:
    print("用法: python script.py [参数]")
    print()
    print("参数类型：")
    print("  1. --key=value    修改单个配置项")
    print("  2. config.py      指定配置文件（Python 格式）")
    print()

    if "config_keys" in globals() and config_keys:
        print("可用的配置项：")
        max_len = max(len(k) for k in config_keys)
        examples = []

        for key in config_keys:
            val = globals().get(key)
            val_type = type(val).__name__ if val is not None else "NoneType"

            # 处理 Path 的显示
            if isinstance(val, Path):
                val_display = f"Path('{str(val)}')"
            else:
                val_display = repr(val)
            if len(val_display) > 40:
                val_display = val_display[:37] + "..."

            print(f"  --{key:<{max_len}} = {val_display:<40} ({val_type})")

            # 生成示例值
            if isinstance(val, bool):
                examples.append(f"--{key}={not val}")
            elif isinstance(val, int):
                examples.append(f"--{key}={val + 100}")
            elif isinstance(val, float):
                examples.append(f"--{key}={round(val + 0.05, 2)}")
            elif isinstance(val, str):
                examples.append(f'--{key}="/new/path"')
            elif isinstance(val, Path):
                examples.append(f'--{key}="/new/path"')
            elif isinstance(val, list):
                examples.append(f"--{key}=\"['item1', 'item2']\"")

        print()
        print("示例：")
        if examples:
            print(f"  python {sys.argv[0]} {examples[0]}")
            if len(examples) > 1:
                print(f"  python {sys.argv[0]} {' '.join(examples[:2])}")
        print(f"  python {sys.argv[0]} my_config.py")
        if any(isinstance(globals().get(k), bool) for k in config_keys):
            bool_key = [k for k in config_keys if isinstance(globals().get(k), bool)][0]
            print(f"  python {sys.argv[0]} --{bool_key}=True")
    else:
        print("  (未检测到配置项)")
    print()
    sys.exit(0)

# 参数解析逻辑
for arg in sys.argv[1:]:
    if "=" not in arg:
        # 配置文件模式
        assert not arg.startswith("--"), f"配置文件名不能以 -- 开头: {arg}"
        config_file = arg
        print(f"Overriding config with {config_file}:")
        exec(open(config_file).read())
    else:
        # --key=value 模式
        assert arg.startswith("--"), f"参数 '{arg}' 格式错误，应为 --key=value 或 config.py"
        key, val = arg.split("=", 1)
        key = key[2:]

        if key not in globals():
            raise ValueError(f"未知配置项: {key}。可用: {list(globals().get('config_keys', []))}")

        original_val = globals()[key]
        original_type = type(original_val)

        # 尝试解析值
        try:
            attempt = literal_eval(val)
        except (SyntaxError, ValueError):
            attempt = val  # 保持字符串

        # 类型转换逻辑
        if original_type == type(attempt):
            # 类型完全匹配，直接使用
            pass
        elif isinstance(original_val, Path) and isinstance(attempt, str):
            # Path 类型自动转换：str -> Path
            attempt = Path(attempt)
        elif isinstance(original_val, bool) and isinstance(attempt, str):
            # Bool 类型特殊处理：处理 "true", "false" 等字符串
            attempt_lower = attempt.lower()
            if attempt_lower in ("true", "1", "yes", "on"):
                attempt = True
            elif attempt_lower in ("false", "0", "no", "off"):
                attempt = False
            else:
                raise ValueError(f"Bool 类型参数 {key} 无法解析: {attempt}。请用 True/False/1/0")
        elif original_val is None:
            # None 默认值，允许任何类型（根据 attempt 的实际类型）
            pass
        else:
            # 类型不匹配，报错
            raise TypeError(f"类型不匹配: {key} 期望 {original_type.__name__}, " f"得到 {type(attempt).__name__} (值: {attempt})。" f'提示: 对于 Path 参数，直接传路径字符串即可，如 --OUTPUT_DIR="/data/output"')

        print(f"Overriding: {key} = {attempt} ({type(attempt).__name__})")
        globals()[key] = attempt
