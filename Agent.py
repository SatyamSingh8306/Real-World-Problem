from ImageDetails import encode_image, analyze_image_with_query
from OCR import PaddleOCRTool

class ImageAnalysisAgent:
    def __init__(self):
        
        self.ocr_tool = PaddleOCRTool()

    def analyze_image(self, image_path, query):
        
        extracted_text = self.ocr_tool.run(image_path)
        encoded_image = encode_image(image_path)
        
        model_name = "llama-3.2-90b-vision-preview"
        llm_analysis_result = analyze_image_with_query(query=query, model=model_name, encoded_image=encoded_image)
        
        # Step 4: Combine the results
        combined_result = {
            "extracted_text": extracted_text,
            "llm_analysis": llm_analysis_result
        }
        return combined_result

if __name__ == "__main__":
    
    image_path = "puzzle.png"
    query = "Acting like a Problem Solver Think and solve the given problem in image."
    agent = ImageAnalysisAgent()
    result = agent.analyze_image(image_path, query)
    
    # Print the results
    print("Extracted Text:", result["extracted_text"])
    print("LLM Analysis:", result["llm_analysis"])