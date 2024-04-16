import PyPDF2

def extract_text_from_pdf(pdf_path):
    with open(pdf_path, 'rb') as file:
        pdf_reader = PyPDF2.PdfReader(file)
        text = ''
        for page in pdf_reader.pages:
            text += page.extract_text() + '\n'  # 添加换行符以便区分不同页面的内容
    return text

# 示例使用
pdf_text = extract_text_from_pdf('input.pdf')
print(pdf_text)

from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont


def create_pdf_with_custom_font(text, font_path, output_pdf_path):
    pdfmetrics.registerFont(TTFont('CustomFont', font_path))

    my_canvas = canvas.Canvas(output_pdf_path, pagesize=letter)
    my_canvas.setFont('CustomFont', 12)  # 设置字体和大小

    # 将文本写入 PDF
    text = text.split('\n')
    y_position = 750
    for line in text:
        my_canvas.drawString(72, y_position, line)
        y_position -= 20  # 移动到下一行

        # 如果接近页面底部，创建新页面
        if y_position < 40:
            my_canvas.showPage()
            my_canvas.setFont('CustomFont', 12)
            y_position = 750

    my_canvas.save()


# 示例使用
create_pdf_with_custom_font(pdf_text, '1.ttf', '1.pdf')
