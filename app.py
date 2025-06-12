import tkinter as tk
from tkinter import ttk, messagebox, filedialog, scrolledtext
import json
import os
from pathlib import Path
import threading
from typing import List, Dict, Any, Optional
from dataclasses import asdict
import webbrowser

# 导入你的FAQ系统
from optimized_faq_system_1 import EnhancedFAQSystem, FAQConfig, FAQItem, FAQVectorizationStrategy

class SimpleFAQApp:
    """面向普通用户的FAQ笔记应用"""
    
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("AI智能问答笔记本")
        self.root.geometry("1200x800")
        self.root.minsize(800, 600)
        
        # 数据存储
        self.data_dir = Path.home() / "AI问答笔记"
        self.data_dir.mkdir(exist_ok=True)
        self.data_file = self.data_dir / "qa_data.json"
        
        # FAQ系统
        self.faq_system = None
        self.qa_items = []
        
        # 界面变量
        self.search_var = tk.StringVar()
        self.current_edit_index = None
        
        self.setup_ui()
        self.load_data()
        self.setup_faq_system()
    
    def setup_ui(self):
        """设置用户界面"""
        # 创建主要框架
        self.create_menu()
        self.create_main_layout()
        self.create_status_bar()
        
        # 设置样式
        self.setup_styles()
    
    def create_menu(self):
        """创建菜单栏"""
        menubar = tk.Menu(self.root)
        self.root.config(menu=menubar)
        
        # 文件菜单
        file_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="文件", menu=file_menu)
        file_menu.add_command(label="导入问答", command=self.import_qa)
        file_menu.add_command(label="导出问答", command=self.export_qa)
        file_menu.add_separator()
        file_menu.add_command(label="退出", command=self.root.quit)
        
        # 编辑菜单
        edit_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="编辑", menu=edit_menu)
        edit_menu.add_command(label="新建问答", command=self.new_qa)
        edit_menu.add_command(label="删除选中", command=self.delete_selected)
        
        # 工具菜单
        tools_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="工具", menu=tools_menu)
        tools_menu.add_command(label="重建搜索索引", command=self.rebuild_index)
        tools_menu.add_command(label="使用帮助", command=self.show_help)
    
    def create_main_layout(self):
        """创建主要布局"""
        # 创建主面板
        main_paned = ttk.PanedWindow(self.root, orient=tk.HORIZONTAL)
        main_paned.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # 左侧：问答列表和搜索
        left_frame = ttk.Frame(main_paned)
        main_paned.add(left_frame, weight=1)
        
        # 右侧：编辑区域
        right_frame = ttk.Frame(main_paned)
        main_paned.add(right_frame, weight=2)
        
        self.create_left_panel(left_frame)
        self.create_right_panel(right_frame)
    
    def create_left_panel(self, parent):
        """创建左侧面板"""
        # 搜索区域
        search_frame = ttk.LabelFrame(parent, text="智能搜索", padding=10)
        search_frame.pack(fill=tk.X, padx=5, pady=5)
        
        search_entry = ttk.Entry(search_frame, textvariable=self.search_var, font=("微软雅黑", 12))
        search_entry.pack(fill=tk.X, pady=(0, 5))
        search_entry.bind('<Return>', lambda e: self.search_qa())
        search_entry.bind('<KeyRelease>', self.on_search_change)
        
        search_btn = ttk.Button(search_frame, text="🔍 搜索", command=self.search_qa)
        search_btn.pack(side=tk.LEFT)
        
        clear_btn = ttk.Button(search_frame, text="清空", command=self.clear_search)
        clear_btn.pack(side=tk.RIGHT)
        
        # 问答列表
        list_frame = ttk.LabelFrame(parent, text="问答列表", padding=5)
        list_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # 创建Treeview
        columns = ("问题", "分类", "相似度")
        self.qa_tree = ttk.Treeview(list_frame, columns=columns, show="headings", height=15)
        
        # 设置列
        self.qa_tree.heading("问题", text="问题")
        self.qa_tree.heading("分类", text="分类")
        self.qa_tree.heading("相似度", text="相似度")
        
        self.qa_tree.column("问题", width=300)
        self.qa_tree.column("分类", width=100)
        self.qa_tree.column("相似度", width=80)
        
        # 滚动条
        scrollbar = ttk.Scrollbar(list_frame, orient=tk.VERTICAL, command=self.qa_tree.yview)
        self.qa_tree.configure(yscrollcommand=scrollbar.set)
        
        self.qa_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # 绑定选择事件
        self.qa_tree.bind('<<TreeviewSelect>>', self.on_item_select)
        self.qa_tree.bind('<Double-1>', self.on_item_double_click)
        
        # 底部按钮
        btn_frame = ttk.Frame(parent)
        btn_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Button(btn_frame, text="➕ 新建", command=self.new_qa).pack(side=tk.LEFT, padx=2)
        ttk.Button(btn_frame, text="✏️ 编辑", command=self.edit_selected).pack(side=tk.LEFT, padx=2)
        ttk.Button(btn_frame, text="🗑️ 删除", command=self.delete_selected).pack(side=tk.LEFT, padx=2)
    
    def create_right_panel(self, parent):
        """创建右侧编辑面板"""
        edit_frame = ttk.LabelFrame(parent, text="编辑问答", padding=10)
        edit_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # 问题输入
        ttk.Label(edit_frame, text="问题：", font=("微软雅黑", 10, "bold")).pack(anchor=tk.W)
        self.question_text = scrolledtext.ScrolledText(edit_frame, height=3, font=("微软雅黑", 11))
        self.question_text.pack(fill=tk.X, pady=(0, 10))
        
        # 答案输入
        ttk.Label(edit_frame, text="答案：", font=("微软雅黑", 10, "bold")).pack(anchor=tk.W)
        self.answer_text = scrolledtext.ScrolledText(edit_frame, height=12, font=("微软雅黑", 11))
        self.answer_text.pack(fill=tk.BOTH, expand=True, pady=(0, 10))
        
        # 分类输入
        category_frame = ttk.Frame(edit_frame)
        category_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Label(category_frame, text="分类：").pack(side=tk.LEFT)
        self.category_var = tk.StringVar()
        self.category_combo = ttk.Combobox(category_frame, textvariable=self.category_var, width=20)
        self.category_combo.pack(side=tk.LEFT, padx=(5, 0))
        
        # 按钮区域
        btn_frame = ttk.Frame(edit_frame)
        btn_frame.pack(fill=tk.X)
        
        ttk.Button(btn_frame, text="💾 保存", command=self.save_qa).pack(side=tk.LEFT, padx=2)
        ttk.Button(btn_frame, text="🔄 清空", command=self.clear_edit).pack(side=tk.LEFT, padx=2)
        ttk.Button(btn_frame, text="❌ 取消", command=self.cancel_edit).pack(side=tk.RIGHT, padx=2)
    
    def create_status_bar(self):
        """创建状态栏"""
        self.status_var = tk.StringVar()
        self.status_var.set("就绪")
        
        status_bar = ttk.Label(self.root, textvariable=self.status_var, relief=tk.SUNKEN)
        status_bar.pack(side=tk.BOTTOM, fill=tk.X)
    
    def setup_styles(self):
        """设置样式"""
        style = ttk.Style()
        
        # 配置Treeview样式
        style.configure("Treeview", font=("微软雅黑", 10))
        style.configure("Treeview.Heading", font=("微软雅黑", 10, "bold"))
    
    def setup_faq_system(self):
        """设置FAQ系统"""
        def setup_in_thread():
            try:
                self.status_var.set("正在初始化AI搜索系统...")
                
                config = FAQConfig(
                    model_name='sentence-transformers/multi-qa-MiniLM-L6-cos-v1',
                    vectorization_strategy=FAQVectorizationStrategy.HYBRID,
                    default_threshold=0.0,
                    show_progress=False
                )
                
                self.faq_system = EnhancedFAQSystem(config)
                
                if self.qa_items:
                    faq_items = [FAQItem(
                        question=item['question'],
                        text=item['answer'],
                        course=item.get('category', ''),
                        metadata=item
                    ) for item in self.qa_items]
                    
                    self.faq_system.load_faq_items(faq_items)
                    self.faq_system.build_system()
                
                self.root.after(0, lambda: self.status_var.set("AI搜索系统已就绪"))
                
            except Exception as e:
                self.root.after(0, lambda: self.status_var.set(f"初始化失败: {str(e)}"))
                self.root.after(0, lambda: messagebox.showerror("错误", f"AI搜索系统初始化失败：\n{str(e)}"))
        
        threading.Thread(target=setup_in_thread, daemon=True).start()
    
    def load_data(self):
        """加载数据"""
        if self.data_file.exists():
            try:
                with open(self.data_file, 'r', encoding='utf-8') as f:
                    self.qa_items = json.load(f)
                self.refresh_list()
                self.update_categories()
                self.status_var.set(f"已加载 {len(self.qa_items)} 个问答")
            except Exception as e:
                messagebox.showerror("错误", f"加载数据失败：\n{str(e)}")
    
    def save_data(self):
        """保存数据"""
        try:
            with open(self.data_file, 'w', encoding='utf-8') as f:
                json.dump(self.qa_items, f, ensure_ascii=False, indent=2)
            return True
        except Exception as e:
            messagebox.showerror("错误", f"保存数据失败：\n{str(e)}")
            return False
    
    def refresh_list(self, items=None):
        """刷新列表显示"""
        # 清空现有项目
        for item in self.qa_tree.get_children():
            self.qa_tree.delete(item)
        
        # 添加项目
        display_items = items if items is not None else self.qa_items
        
        for i, item in enumerate(display_items):
            question = item['question'][:50] + "..." if len(item['question']) > 50 else item['question']
            category = item.get('category', '未分类')
            similarity = f"{item.get('similarity', 1.0):.3f}" if 'similarity' in item else ""
            
            self.qa_tree.insert('', tk.END, values=(question, category, similarity), tags=(str(i),))
    
    def update_categories(self):
        """更新分类列表"""
        categories = list(set(item.get('category', '未分类') for item in self.qa_items))
        categories.sort()
        self.category_combo['values'] = categories
    
    def search_qa(self):
        """搜索问答"""
        query = self.search_var.get().strip()
        if not query:
            self.refresh_list()
            return
        
        if not self.faq_system:
            messagebox.showwarning("提示", "AI搜索系统尚未就绪，请稍候再试")
            return
        
        def search_in_thread():
            try:
                self.root.after(0, lambda: self.status_var.set("正在搜索..."))
                
                results = self.faq_system.search(query, top_k=20, threshold=0.0)
                
                # 转换结果格式
                search_results = []
                for result in results:
                    faq_item = result['faq_item']
                    original_item = None
                    
                    # 找到原始数据项
                    for item in self.qa_items:
                        if (item['question'] == faq_item.question and 
                            item['answer'] == faq_item.text):
                            original_item = item.copy()
                            break
                    
                    if original_item:
                        original_item['similarity'] = result['similarity']
                        search_results.append(original_item)
                
                self.root.after(0, lambda: self.refresh_list(search_results))
                self.root.after(0, lambda: self.status_var.set(f"找到 {len(search_results)} 个相关结果"))
                
            except Exception as e:
                self.root.after(0, lambda: self.status_var.set("搜索失败"))
                self.root.after(0, lambda: messagebox.showerror("错误", f"搜索失败：\n{str(e)}"))
        
        threading.Thread(target=search_in_thread, daemon=True).start()
    
    def on_search_change(self, event):
        """搜索内容改变时的处理"""
        # 如果搜索框为空，显示所有项目
        if not self.search_var.get().strip():
            self.refresh_list()
    
    def clear_search(self):
        """清空搜索"""
        self.search_var.set("")
        self.refresh_list()
        self.status_var.set("已清空搜索")
    
    def on_item_select(self, event):
        """选择项目时的处理"""
        selection = self.qa_tree.selection()
        if selection:
            item = self.qa_tree.item(selection[0])
            index = int(item['tags'][0])
            
            # 查找对应的原始数据
            if 'similarity' in self.qa_items[0]:  # 如果是搜索结果
                # 需要在原始数据中找到对应项
                display_items = [item for item in self.qa_tree.get_children()]
                if index < len(display_items):
                    values = self.qa_tree.item(display_items[index])['values']
                    question_preview = values[0]
                    
                    for i, original_item in enumerate(self.qa_items):
                        if original_item['question'].startswith(question_preview.replace("...", "")):
                            self.show_item_preview(original_item)
                            break
            else:
                if index < len(self.qa_items):
                    self.show_item_preview(self.qa_items[index])
    
    def show_item_preview(self, item):
        """显示项目预览"""
        # 在编辑区域显示但不允许编辑
        self.question_text.delete(1.0, tk.END)
        self.question_text.insert(1.0, item['question'])
        
        self.answer_text.delete(1.0, tk.END)
        self.answer_text.insert(1.0, item['answer'])
        
        self.category_var.set(item.get('category', ''))
    
    def on_item_double_click(self, event):
        """双击编辑"""
        self.edit_selected()
    
    def new_qa(self):
        """新建问答"""
        self.current_edit_index = None
        self.clear_edit()
        self.question_text.focus()
    
    def edit_selected(self):
        """编辑选中项"""
        selection = self.qa_tree.selection()
        if not selection:
            messagebox.showwarning("提示", "请先选择要编辑的问答")
            return
        
        item = self.qa_tree.item(selection[0])
        index = int(item['tags'][0])
        
        if index < len(self.qa_items):
            self.current_edit_index = index
            qa_item = self.qa_items[index]
            
            self.question_text.delete(1.0, tk.END)
            self.question_text.insert(1.0, qa_item['question'])
            
            self.answer_text.delete(1.0, tk.END)
            self.answer_text.insert(1.0, qa_item['answer'])
            
            self.category_var.set(qa_item.get('category', ''))
    
    def save_qa(self):
        """保存问答"""
        question = self.question_text.get(1.0, tk.END).strip()
        answer = self.answer_text.get(1.0, tk.END).strip()
        category = self.category_var.get().strip()
        
        if not question:
            messagebox.showwarning("提示", "请输入问题")
            return
        
        if not answer:
            messagebox.showwarning("提示", "请输入答案")
            return
        
        qa_item = {
            'question': question,
            'answer': answer,
            'category': category or '未分类'
        }
        
        if self.current_edit_index is not None:
            # 编辑现有项
            self.qa_items[self.current_edit_index] = qa_item
            action = "更新"
        else:
            # 新建项
            self.qa_items.append(qa_item)
            action = "添加"
        
        if self.save_data():
            self.refresh_list()
            self.update_categories()
            self.clear_edit()
            self.status_var.set(f"已{action}问答，正在重建搜索索引...")
            
            # 重建FAQ系统
            self.setup_faq_system()
    
    def delete_selected(self):
        """删除选中项"""
        selection = self.qa_tree.selection()
        if not selection:
            messagebox.showwarning("提示", "请先选择要删除的问答")
            return
        
        if messagebox.askyesno("确认", "确定要删除选中的问答吗？"):
            item = self.qa_tree.item(selection[0])
            index = int(item['tags'][0])
            
            if index < len(self.qa_items):
                del self.qa_items[index]
                
                if self.save_data():
                    self.refresh_list()
                    self.update_categories()
                    self.clear_edit()
                    self.status_var.set("已删除问答，正在重建搜索索引...")
                    self.setup_faq_system()
    
    def clear_edit(self):
        """清空编辑区域"""
        self.question_text.delete(1.0, tk.END)
        self.answer_text.delete(1.0, tk.END)
        self.category_var.set("")
        self.current_edit_index = None
    
    def cancel_edit(self):
        """取消编辑"""
        self.clear_edit()
        self.status_var.set("已取消编辑")
    
    def rebuild_index(self):
        """重建搜索索引"""
        self.setup_faq_system()
        messagebox.showinfo("提示", "搜索索引重建完成")
    
    def import_qa(self):
        """导入问答"""
        filename = filedialog.askopenfilename(
            title="导入问答文件",
            filetypes=[("JSON文件", "*.json"), ("所有文件", "*.*")]
        )
        
        if filename:
            try:
                with open(filename, 'r', encoding='utf-8') as f:
                    imported_data = json.load(f)
                
                if isinstance(imported_data, list):
                    self.qa_items.extend(imported_data)
                    if self.save_data():
                        self.refresh_list()
                        self.update_categories()
                        self.setup_faq_system()
                        messagebox.showinfo("成功", f"已导入 {len(imported_data)} 个问答")
                
            except Exception as e:
                messagebox.showerror("错误", f"导入失败：\n{str(e)}")
    
    def export_qa(self):
        """导出问答"""
        filename = filedialog.asksaveasfilename(
            title="导出问答文件",
            defaultextension=".json",
            filetypes=[("JSON文件", "*.json"), ("所有文件", "*.*")]
        )
        
        if filename:
            try:
                with open(filename, 'w', encoding='utf-8') as f:
                    json.dump(self.qa_items, f, ensure_ascii=False, indent=2)
                messagebox.showinfo("成功", f"已导出 {len(self.qa_items)} 个问答")
            except Exception as e:
                messagebox.showerror("错误", f"导出失败：\n{str(e)}")
    
    def show_help(self):
        """显示帮助"""
        help_text = """
AI智能问答笔记本 - 使用帮助

📝 基本功能：
• 新建问答：点击"新建"按钮，输入问题和答案，点击"保存"
• 编辑问答：双击列表中的项目，或选中后点击"编辑"
• 删除问答：选中项目后点击"删除"

🔍 智能搜索：
• 在搜索框输入关键词，系统会智能匹配相关问答
• 搜索结果按相似度排序，数值越高越相关
• 清空搜索框可显示所有问答

📁 分类管理：
• 为问答设置分类，便于组织和管理
• 分类会自动出现在下拉列表中

💾 数据管理：
• 数据自动保存在用户文件夹的"AI问答笔记"目录
• 支持导入/导出JSON格式文件
• 可重建搜索索引提升搜索效果

🎯 使用技巧：
• 问题表述要清晰具体
• 答案可以包含详细信息
• 合理使用分类便于管理
• 定期备份数据文件
        """
        
        help_window = tk.Toplevel(self.root)
        help_window.title("使用帮助")
        help_window.geometry("600x500")
        help_window.resizable(False, False)
        
        text_widget = scrolledtext.ScrolledText(help_window, font=("微软雅黑", 10))
        text_widget.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        text_widget.insert(1.0, help_text)
        text_widget.config(state=tk.DISABLED)
    
    def run(self):
        """运行应用"""
        self.root.mainloop()


def main():
    """主函数"""
    app = SimpleFAQApp()
    app.run()


if __name__ == "__main__":
    main()