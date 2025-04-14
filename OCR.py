from langchain.tools import BaseTool
from paddleocr import PaddleOCR
from pydantic import Field

class PaddleOCRTool(BaseTool):
    name: str = "PaddleOCR Tool"
    description: str = "Extracts text from images using PaddleOCR"

    
    ocr: PaddleOCR = Field(default=None)

    def __init__(self):
        super().__init__()
        self.ocr = PaddleOCR(use_angle_cls=True, lang='en')  

    def _run(self, image_path: str) -> str:
        
        result = self.ocr.ocr(image_path, cls=True)
        extracted_text = "\n".join([line[1][0] for line in result[0]])
        return extracted_text

    def _arun(self, image_path: str):
        raise NotImplementedError("This tool does not support async")
    

if __name__ == "__main__":
    ocr_tool = PaddleOCRTool()
    extracted_text = ocr_tool.run("trail.jpeg")  
    print("Extracted Text:", extracted_text)