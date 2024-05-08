import pikepdf

PATH = "data/plan/3daybeginners.pdf"

pdf = pikepdf.Pdf.open(PATH)

pdf_metadata = pdf.docinfo

for key, value in pdf_metadata.items():
    print(f'{key} : {value}')
