"""
PSD Channel Packer
读取 PSD 顶层结构，将指定图层/组的通道打包到输出图的 RGBA 中。

两种模式：
- RGB 颜色模式：取该层 RGB → 输出 RGB（占 3 通道），只能有一层用
- 单通道模式：取该层的 R/G/B/A 其一 → 放到输出的 R/G/B/A 其一
"""

import os
import tkinter as tk
from pathlib import Path

import customtkinter as ctk
import numpy as np
from PIL import Image
from psd_tools import PSDImage

# ── 颜色/字体常量 ──
_BG_WINDOW = "#1a1a1a"
_BG_CARD = "#2b2b2b"
_BG_CARD_INNER = "#333333"
_FG_TEXT = "#e0e0e0"
_FG_DIM = "#888888"
_ACCENT = "#e59639"
_ACCENT_HOVER = "#c07d2e"
_FONT = ("Microsoft YaHei UI", 11)
_FONT_BOLD = ("Microsoft YaHei UI", 11, "bold")
_FONT_TITLE = ("Microsoft YaHei UI", 12, "bold")

# 模式选项
MODE_OPTIONS = ["不用", "RGB颜色", "单通道"]
CHANNEL_OPTIONS = ["R", "G", "B", "A"]


def composite_layer(layer) -> np.ndarray:
    """合并图层/组为 RGBA numpy 数组"""
    if layer.is_group():
        img = layer.composite()
    else:
        img = layer.topil()
    if img is None:
        return None
    return np.array(img.convert('RGBA'))


def pack_channels(assignments: list, psd_layers: list, output_size: tuple) -> np.ndarray:
    """
    根据分配方案打包通道。
    
    assignments: [(layer_idx, mode, src_channel, dst_channel), ...]
    Returns: (H, W, 4) uint8 RGBA
    """
    H, W = output_size
    output = np.zeros((H, W, 4), dtype=np.uint8)

    for layer_idx, mode, src_ch, dst_ch in assignments:
        if mode == 'skip':
            continue

        layer = psd_layers[layer_idx]
        arr = composite_layer(layer)
        if arr is None:
            continue

        pil_img = Image.fromarray(arr, mode='RGBA')
        pil_img = pil_img.resize((W, H), Image.LANCZOS)
        arr = np.array(pil_img)

        if mode == 'rgb':
            output[..., 0] = arr[..., 0]
            output[..., 1] = arr[..., 1]
            output[..., 2] = arr[..., 2]
        elif mode == 'single':
            src_idx = {'R': 0, 'G': 1, 'B': 2, 'A': 3}[src_ch]
            dst_idx = {'R': 0, 'G': 1, 'B': 2, 'A': 3}[dst_ch]
            output[..., dst_idx] = arr[..., src_idx]

    return output


def run_gui():
    ctk.set_appearance_mode("dark")
    ctk.set_default_color_theme("dark-blue")

    root = ctk.CTk()
    root.title("PSD Channel Packer + AI Mask")
    root.geometry("780x750")
    root.minsize(720, 650)
    root.configure(fg_color=_BG_WINDOW)

    # ── Tabview ──
    tabview = ctk.CTkTabview(root, fg_color=_BG_WINDOW)
    tabview.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
    tab_ai = tabview.add("AI Mask 生成")
    tab_packer = tabview.add("Channel Packer")

    # ══════════════════════════════════════════════════════
    # TAB 1: Channel Packer
    # ══════════════════════════════════════════════════════

    # ── 状态 ──
    psd_obj = [None]
    psd_layers = []
    # layer_widgets: list of dicts with keys: mode_var, src_var, dst_var, src_combo, dst_combo, mode_combo
    layer_widgets = []

    # ── 变量 ──
    var_psd_path = tk.StringVar()
    var_size_mode = tk.StringVar(value="原始尺寸")
    var_width = tk.StringVar(value="")
    var_height = tk.StringVar(value="")
    var_format = tk.StringVar(value="PNG")
    var_output_path = tk.StringVar()

    # ── 滚动主体（Channel Packer tab）──
    main_scroll = ctk.CTkScrollableFrame(tab_packer, fg_color=_BG_WINDOW)
    main_scroll.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

    # ════════════════════════════════════════════
    # CARD 1: PSD 文件
    # ════════════════════════════════════════════
    card_file = ctk.CTkFrame(main_scroll, fg_color=_BG_CARD, corner_radius=10)
    card_file.pack(fill=tk.X, pady=(0, 12))

    ctk.CTkLabel(card_file, text="PSD 文件", font=_FONT_TITLE, text_color=_FG_TEXT).pack(
        anchor=tk.W, padx=15, pady=(10, 5))

    row_psd = ctk.CTkFrame(card_file, fg_color="transparent")
    row_psd.pack(fill=tk.X, padx=15, pady=(0, 10))
    ctk.CTkEntry(row_psd, textvariable=var_psd_path, font=_FONT, height=28).pack(
        side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 8))

    def browse_psd():
        from tkinter import filedialog
        path = filedialog.askopenfilename(filetypes=[("PSD files", "*.psd"), ("All", "*.*")])
        if path:
            var_psd_path.set(path)
            load_psd()

    ctk.CTkButton(row_psd, text="浏览", width=60, font=_FONT, command=browse_psd).pack(side=tk.LEFT, padx=(0, 8))

    def load_psd():
        path = var_psd_path.get()
        if not path or not os.path.exists(path):
            return
        psd_obj[0] = PSDImage.open(path)
        psd_layers.clear()
        for layer in psd_obj[0]:
            psd_layers.append(layer)
        populate_layers()
        # 显示原始尺寸
        var_width.set(str(psd_obj[0].width))
        var_height.set(str(psd_obj[0].height))
        # 自动输出路径: 原文件名_DO + 当前格式后缀
        update_output_path()

    ctk.CTkButton(row_psd, text="加载", width=60, font=_FONT,
                  fg_color=_ACCENT, hover_color=_ACCENT_HOVER, command=load_psd).pack(side=tk.LEFT)

    # ════════════════════════════════════════════
    # CARD 2: 图层分配
    # ════════════════════════════════════════════
    card_layers = ctk.CTkFrame(main_scroll, fg_color=_BG_CARD, corner_radius=10)
    card_layers.pack(fill=tk.X, pady=(0, 12))

    ctk.CTkLabel(card_layers, text="图层通道分配", font=_FONT_TITLE, text_color=_FG_TEXT).pack(
        anchor=tk.W, padx=15, pady=(10, 5))

    layers_frame = ctk.CTkScrollableFrame(card_layers, fg_color=_BG_CARD_INNER, corner_radius=6, height=200)
    layers_frame.pack(fill=tk.X, padx=15, pady=(0, 10))

    def has_rgb_layer():
        """检查是否已有层设为 RGB 模式"""
        for w in layer_widgets:
            if w['mode_var'].get() == "RGB颜色":
                return True
        return False

    def update_dst_constraints():
        """当有 RGB 层时，其他单通道层的输出只能选 A"""
        rgb_exists = has_rgb_layer()
        for w in layer_widgets:
            if w['mode_var'].get() == "RGB颜色":
                continue
            if w['mode_var'].get() == "单通道":
                if rgb_exists:
                    w['dst_combo'].configure(values=["A"])
                    w['dst_var'].set("A")
                else:
                    w['dst_combo'].configure(values=CHANNEL_OPTIONS)

    def populate_layers():
        for w in layers_frame.winfo_children():
            w.destroy()
        layer_widgets.clear()

        if not psd_layers:
            ctk.CTkLabel(layers_frame, text="加载 PSD 后显示图层", font=_FONT,
                         text_color=_FG_DIM).pack(pady=10)
            return

        # Header
        hdr = ctk.CTkFrame(layers_frame, fg_color="transparent")
        hdr.pack(fill=tk.X, padx=10, pady=(8, 4))
        ctk.CTkLabel(hdr, text="图层名", font=_FONT_BOLD, width=180, anchor="w").pack(side=tk.LEFT)
        ctk.CTkLabel(hdr, text="模式", font=_FONT_BOLD, width=100).pack(side=tk.LEFT, padx=(10, 0))
        ctk.CTkLabel(hdr, text="源通道", font=_FONT_BOLD, width=70).pack(side=tk.LEFT, padx=(10, 0))
        ctk.CTkLabel(hdr, text="→ 输出", font=_FONT_BOLD, width=70).pack(side=tk.LEFT, padx=(10, 0))

        for i, layer in enumerate(psd_layers):
            row = ctk.CTkFrame(layers_frame, fg_color="transparent")
            row.pack(fill=tk.X, padx=10, pady=2)

            icon = "📁" if layer.is_group() else "📄"
            name = f"{icon} {layer.name}"
            ctk.CTkLabel(row, text=name, font=_FONT, width=180, anchor="w").pack(side=tk.LEFT)

            mode_var = tk.StringVar(value="不用")
            src_var = tk.StringVar(value="A")
            dst_var = tk.StringVar(value="A")

            mode_combo = ctk.CTkComboBox(row, variable=mode_var, values=MODE_OPTIONS,
                                         width=100, font=_FONT, height=26)
            mode_combo.pack(side=tk.LEFT, padx=(10, 0))

            src_combo = ctk.CTkComboBox(row, variable=src_var, values=CHANNEL_OPTIONS,
                                        width=70, font=_FONT, height=26, state="disabled")
            src_combo.pack(side=tk.LEFT, padx=(10, 0))

            dst_combo = ctk.CTkComboBox(row, variable=dst_var, values=CHANNEL_OPTIONS,
                                        width=70, font=_FONT, height=26, state="disabled")
            dst_combo.pack(side=tk.LEFT, padx=(10, 0))

            widget_dict = {
                'mode_var': mode_var, 'src_var': src_var, 'dst_var': dst_var,
                'mode_combo': mode_combo, 'src_combo': src_combo, 'dst_combo': dst_combo,
            }
            layer_widgets.append(widget_dict)

            def on_mode_change(choice, w=widget_dict):
                mode = w['mode_var'].get()
                if mode == "单通道":
                    w['src_combo'].configure(state="normal", values=CHANNEL_OPTIONS)
                    w['dst_combo'].configure(state="normal")
                    # 重置为合理默认值
                    if w['src_var'].get() not in CHANNEL_OPTIONS:
                        w['src_var'].set("A")
                    if w['dst_var'].get() not in CHANNEL_OPTIONS:
                        w['dst_var'].set("A")
                elif mode == "RGB颜色":
                    w['src_var'].set("RGB")
                    w['dst_var'].set("RGB")
                    w['src_combo'].configure(state="disabled", values=["RGB"])
                    w['dst_combo'].configure(state="disabled", values=["RGB"])
                else:  # 不用
                    w['src_combo'].configure(state="disabled")
                    w['dst_combo'].configure(state="disabled")
                # 更新所有层的约束
                update_dst_constraints()

            mode_combo.configure(command=on_mode_change)

    populate_layers()

    # ════════════════════════════════════════════
    # CARD 3: 输出设置
    # ════════════════════════════════════════════
    card_output = ctk.CTkFrame(main_scroll, fg_color=_BG_CARD, corner_radius=10)
    card_output.pack(fill=tk.X, pady=(0, 12))

    ctk.CTkLabel(card_output, text="输出设置", font=_FONT_TITLE, text_color=_FG_TEXT).pack(
        anchor=tk.W, padx=15, pady=(10, 5))

    # 尺寸
    row_size = ctk.CTkFrame(card_output, fg_color="transparent")
    row_size.pack(fill=tk.X, padx=15, pady=3)
    ctk.CTkLabel(row_size, text="尺寸", font=_FONT, width=60).pack(side=tk.LEFT)

    size_combo = ctk.CTkComboBox(row_size, variable=var_size_mode,
                    values=["原始尺寸", "等比缩放", "自定义"], width=120, font=_FONT, height=28)
    size_combo.pack(side=tk.LEFT, padx=(10, 10))

    width_entry = ctk.CTkEntry(row_size, textvariable=var_width, font=_FONT, height=28, width=70,
                               state="disabled")
    width_entry.pack(side=tk.LEFT)
    ctk.CTkLabel(row_size, text="x", font=_FONT).pack(side=tk.LEFT, padx=5)
    # 高度用 label 显示（等比缩放时只读）
    height_entry = ctk.CTkEntry(row_size, textvariable=var_height, font=_FONT, height=28, width=70,
                                state="disabled")
    height_entry.pack(side=tk.LEFT)

    def calc_height_from_width():
        """等比缩放模式下，根据宽度实时计算高度"""
        if not psd_obj[0]:
            return
        if var_size_mode.get() != "等比缩放":
            return
        try:
            w = int(var_width.get())
            orig_w, orig_h = psd_obj[0].width, psd_obj[0].height
            h = int(orig_h * w / orig_w)
            var_height.set(str(h))
        except (ValueError, ZeroDivisionError):
            pass

    def on_size_mode_change(choice):
        if choice == "原始尺寸":
            width_entry.configure(state="disabled")
            height_entry.configure(state="disabled")
            # 显示原始尺寸
            if psd_obj[0]:
                var_width.set(str(psd_obj[0].width))
                var_height.set(str(psd_obj[0].height))
        elif choice == "等比缩放":
            width_entry.configure(state="normal")
            height_entry.configure(state="disabled")
            calc_height_from_width()
        else:  # 自定义
            width_entry.configure(state="normal")
            height_entry.configure(state="normal")

    size_combo.configure(command=on_size_mode_change)

    # 宽度变化时实时算高度（等比缩放模式）
    var_width.trace_add('write', lambda *_: calc_height_from_width())

    # 格式
    row_fmt = ctk.CTkFrame(card_output, fg_color="transparent")
    row_fmt.pack(fill=tk.X, padx=15, pady=3)
    ctk.CTkLabel(row_fmt, text="格式", font=_FONT, width=60).pack(side=tk.LEFT)

    def on_format_change(choice):
        """格式切换时自动更新输出路径后缀"""
        path = var_output_path.get()
        if path:
            ext = ".png" if choice == "PNG" else ".tga"
            var_output_path.set(str(Path(path).with_suffix(ext)))

    fmt_combo = ctk.CTkComboBox(row_fmt, variable=var_format, values=["PNG", "TGA"],
                    width=100, font=_FONT, height=28, command=on_format_change)
    fmt_combo.pack(side=tk.LEFT, padx=(10, 0))

    # 输出路径
    row_out = ctk.CTkFrame(card_output, fg_color="transparent")
    row_out.pack(fill=tk.X, padx=15, pady=(3, 10))
    ctk.CTkLabel(row_out, text="输出", font=_FONT, width=60).pack(side=tk.LEFT)
    ctk.CTkEntry(row_out, textvariable=var_output_path, font=_FONT, height=28).pack(
        side=tk.LEFT, fill=tk.X, expand=True, padx=(10, 8))

    def browse_output():
        from tkinter import filedialog
        ext = ".png" if var_format.get() == "PNG" else ".tga"
        path = filedialog.asksaveasfilename(defaultextension=ext,
                                            filetypes=[(f"{ext[1:].upper()} files", f"*{ext}")])
        if path:
            var_output_path.set(path)

    ctk.CTkButton(row_out, text="...", width=40, font=_FONT, command=browse_output).pack(side=tk.LEFT)

    def update_output_path():
        """根据 PSD 路径和当前格式生成默认输出路径"""
        path = var_psd_path.get()
        if path:
            stem = Path(path).stem
            parent = Path(path).parent
            ext = ".png" if var_format.get() == "PNG" else ".tga"
            var_output_path.set(str(parent / f"{stem}_DO{ext}"))

    # ════════════════════════════════════════════
    # 导出按钮
    # ════════════════════════════════════════════
    card_action = ctk.CTkFrame(main_scroll, fg_color=_BG_CARD, corner_radius=10)
    card_action.pack(fill=tk.X, pady=(0, 12))

    status_label = ctk.CTkLabel(card_action, text="", font=_FONT, text_color=_FG_DIM)
    status_label.pack(anchor=tk.W, padx=15, pady=(10, 0))

    def do_export():
        if not psd_obj[0] or not psd_layers:
            status_label.configure(text="请先加载 PSD", text_color="#d44")
            return

        # 确定输出尺寸
        orig_w, orig_h = psd_obj[0].width, psd_obj[0].height
        size_mode = var_size_mode.get()
        if size_mode == "原始尺寸":
            out_w, out_h = orig_w, orig_h
        elif size_mode == "等比缩放":
            try:
                out_w = int(var_width.get())
            except ValueError:
                out_w = orig_w
            scale = out_w / orig_w
            out_h = int(orig_h * scale)
        else:  # 自定义
            try:
                out_w = int(var_width.get())
                out_h = int(var_height.get())
            except ValueError:
                status_label.configure(text="尺寸输入无效", text_color="#d44")
                return

        # 构建分配方案 + 校验
        assignments = []
        has_rgb = False
        has_any = False
        used_dst = set()

        for i, w in enumerate(layer_widgets):
            mode_str = w['mode_var'].get()
            if mode_str == "不用":
                assignments.append((i, 'skip', '', ''))
            elif mode_str == "RGB颜色":
                has_rgb = True
                has_any = True
                assignments.append((i, 'rgb', '', ''))
                used_dst.update(['R', 'G', 'B'])
            elif mode_str == "单通道":
                src = w['src_var'].get()
                dst = w['dst_var'].get()
                if dst in used_dst:
                    status_label.configure(
                        text=f"通道冲突: {dst} 通道被多层占用", text_color="#d44")
                    return
                used_dst.add(dst)
                has_any = True
                assignments.append((i, 'single', src, dst))

        if not has_any:
            status_label.configure(text="至少需要配置一个输出通道", text_color="#d44")
            return

        # 打包
        status_label.configure(text="导出中...", text_color=_FG_DIM)
        root.update()
        output = pack_channels(assignments, psd_layers, (out_h, out_w))

        # 保存
        out_path = var_output_path.get()
        if not out_path:
            status_label.configure(text="请指定输出路径", text_color="#d44")
            return

        fmt = var_format.get()
        ext = ".png" if fmt == "PNG" else ".tga"
        out_path = str(Path(out_path).with_suffix(ext))
        img = Image.fromarray(output, mode='RGBA')
        img.save(out_path)

        status_label.configure(text=f"导出成功: {out_path} ({out_w}x{out_h})", text_color="#52A852")

    btn_row = ctk.CTkFrame(card_action, fg_color="transparent")
    btn_row.pack(fill=tk.X, padx=15, pady=(5, 10))
    ctk.CTkButton(btn_row, text="导出", font=_FONT_BOLD, width=140,
                  fg_color=_ACCENT, hover_color=_ACCENT_HOVER, command=do_export).pack(side=tk.LEFT)

    # ══════════════════════════════════════════════════════
    # TAB 2: AI Mask 生成
    # ══════════════════════════════════════════════════════
    ai_scroll = ctk.CTkScrollableFrame(tab_ai, fg_color=_BG_WINDOW)
    ai_scroll.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

    # 变量
    var_ai_input = tk.StringVar()
    var_ai_output = tk.StringVar()
    var_ai_prompt = tk.StringVar(value="提取这张图的水面mask。保持原图透视，水面为白色，其余部分为黑色，靠近岸边的地方要有过渡")
    var_ai_layer_name = tk.StringVar(value="AI_mask")

    # 输入图片
    card_ai_input = ctk.CTkFrame(ai_scroll, fg_color=_BG_CARD, corner_radius=10)
    card_ai_input.pack(fill=tk.X, pady=(0, 12))
    ctk.CTkLabel(card_ai_input, text="AI Mask 生成", font=_FONT_TITLE, text_color=_FG_TEXT).pack(
        anchor=tk.W, padx=15, pady=(10, 5))

    ctk.CTkLabel(card_ai_input, text="输入一张图 → AI 生成 mask → 原图+mask 一起存为 PSD",
                 font=_FONT, text_color=_FG_DIM).pack(anchor=tk.W, padx=15, pady=(0, 8))

    row_ai_img = ctk.CTkFrame(card_ai_input, fg_color="transparent")
    row_ai_img.pack(fill=tk.X, padx=15, pady=3)
    ctk.CTkLabel(row_ai_img, text="输入图片", font=_FONT, width=80).pack(side=tk.LEFT)
    ctk.CTkEntry(row_ai_img, textvariable=var_ai_input, font=_FONT, height=28).pack(
        side=tk.LEFT, fill=tk.X, expand=True, padx=(10, 8))

    def browse_ai_input():
        from tkinter import filedialog
        path = filedialog.askopenfilename(filetypes=[("Image", "*.png *.jpg *.jpeg"), ("All", "*.*")])
        if path:
            var_ai_input.set(path)
            # 自动生成输出路径
            stem = Path(path).stem
            parent = Path(path).parent
            var_ai_output.set(str(parent / f"{stem}_DO.psd"))

    ctk.CTkButton(row_ai_img, text="...", width=40, font=_FONT, command=browse_ai_input).pack(side=tk.LEFT)

    # Prompt
    row_ai_prompt = ctk.CTkFrame(card_ai_input, fg_color="transparent")
    row_ai_prompt.pack(fill=tk.X, padx=15, pady=3)
    ctk.CTkLabel(row_ai_prompt, text="Prompt", font=_FONT, width=80).pack(side=tk.LEFT)
    ctk.CTkEntry(row_ai_prompt, textvariable=var_ai_prompt, font=_FONT, height=28).pack(
        side=tk.LEFT, fill=tk.X, expand=True, padx=(10, 0))

    # 图层名
    row_ai_name = ctk.CTkFrame(card_ai_input, fg_color="transparent")
    row_ai_name.pack(fill=tk.X, padx=15, pady=3)
    ctk.CTkLabel(row_ai_name, text="Mask 图层名", font=_FONT, width=80).pack(side=tk.LEFT)
    ctk.CTkEntry(row_ai_name, textvariable=var_ai_layer_name, font=_FONT, height=28, width=200).pack(
        side=tk.LEFT, padx=(10, 0))

    # 输出路径
    row_ai_out = ctk.CTkFrame(card_ai_input, fg_color="transparent")
    row_ai_out.pack(fill=tk.X, padx=15, pady=(3, 10))
    ctk.CTkLabel(row_ai_out, text="输出 PSD", font=_FONT, width=80).pack(side=tk.LEFT)
    ctk.CTkEntry(row_ai_out, textvariable=var_ai_output, font=_FONT, height=28).pack(
        side=tk.LEFT, fill=tk.X, expand=True, padx=(10, 8))

    def browse_ai_output():
        from tkinter import filedialog
        path = filedialog.asksaveasfilename(defaultextension=".psd",
                                            filetypes=[("PSD files", "*.psd")])
        if path:
            var_ai_output.set(path)

    ctk.CTkButton(row_ai_out, text="...", width=40, font=_FONT, command=browse_ai_output).pack(side=tk.LEFT)

    # 状态 + 按钮
    card_ai_action = ctk.CTkFrame(ai_scroll, fg_color=_BG_CARD, corner_radius=10)
    card_ai_action.pack(fill=tk.X, pady=(0, 12))

    ai_status = ctk.CTkLabel(card_ai_action, text="", font=_FONT, text_color=_FG_DIM)
    ai_status.pack(anchor=tk.W, padx=15, pady=(10, 0))

    ai_btn_row = ctk.CTkFrame(card_ai_action, fg_color="transparent")
    ai_btn_row.pack(fill=tk.X, padx=15, pady=(5, 10))

    def do_ai_generate():
        import threading
        from ai_mask_gen import generate_mask_and_create_psd

        input_img = var_ai_input.get()
        output_psd = var_ai_output.get()
        prompt = var_ai_prompt.get()
        layer_name = var_ai_layer_name.get() or "AI_mask"

        if not input_img or not os.path.exists(input_img):
            ai_status.configure(text="请选择有效的输入图片", text_color="#d44")
            return
        if not prompt:
            ai_status.configure(text="请填写 Prompt", text_color="#d44")
            return
        if not output_psd:
            ai_status.configure(text="请指定输出 PSD 路径", text_color="#d44")
            return

        ai_btn.configure(state="disabled")
        ai_status.configure(text="AI 生成中...", text_color=_FG_DIM)

        def worker():
            def update_status(msg, color=_FG_DIM):
                root.after(0, lambda: ai_status.configure(text=msg, text_color=color))

            try:
                output = generate_mask_and_create_psd(
                    input_image_path=input_img,
                    prompt=prompt,
                    output_psd_path=output_psd,
                    layer_name=layer_name,
                    progress_cb=lambda msg: update_status(msg),
                )
                update_status(f"完成! PSD: {output}", "#52A852")
            except Exception as e:
                update_status(f"错误: {e}", "#d44")
            finally:
                root.after(0, lambda: ai_btn.configure(state="normal"))

        threading.Thread(target=worker, daemon=True).start()

    ai_btn = ctk.CTkButton(ai_btn_row, text="生成 Mask 并写入 PSD", font=_FONT_BOLD, width=200,
                           fg_color=_ACCENT, hover_color=_ACCENT_HOVER, command=do_ai_generate)
    ai_btn.pack(side=tk.LEFT)

    root.mainloop()


if __name__ == "__main__":
    run_gui()
