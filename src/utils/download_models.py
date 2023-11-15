from transformers import VisionEncoderDecoderModel
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments
from transformers import TrOCRProcessor
from transformers import AutoTokenizer
from transformers import ViTFeatureExtractor

#processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten")
#model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-stage1")

from transformers import VisionEncoderDecoderModel
from transformers import TrOCRProcessor

def load_processor() -> TrOCRProcessor:
    feature_extractor=ViTFeatureExtractor.from_pretrained("google/vit-base-patch16-224-in21k")
    model_path = "Riksarkivet/bert-base-cased-swe-historical"
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    return TrOCRProcessor(feature_extractor=feature_extractor, tokenizer=tokenizer)

#processor = load_processor()
model = VisionEncoderDecoderModel.from_encoder_decoder_pretrained("google/vit-base-patch16-224", "Riksarkivet/bert-base-cased-swe-historical")