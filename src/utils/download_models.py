from transformers import VisionEncoderDecoderModel
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments
from transformers import TrOCRProcessor
from transformers import AutoTokenizer
from transformers import ViTFeatureExtractor
import evaluate

#processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten")
#model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-stage1")

from transformers import VisionEncoderDecoderModel
from transformers import TrOCRProcessor

def load_processor() -> TrOCRProcessor:
    feature_extractor=ViTFeatureExtractor.from_pretrained("google/vit-base-patch16-224", cache_dir='/leonardo/home/userexternal/elenas00/projects/huggingface_htr/models/hub')
    model_path = "Riksarkivet/bert-base-cased-swe-historical"
    tokenizer = AutoTokenizer.from_pretrained(model_path, cache_dir='/leonardo/home/userexternal/elenas00/projects/huggingface_htr/models/hub')
    return TrOCRProcessor(feature_extractor=feature_extractor, tokenizer=tokenizer)

processor = load_processor()
model = VisionEncoderDecoderModel.from_encoder_decoder_pretrained("google/vit-base-patch16-224", 
                                                                "Riksarkivet/bert-base-cased-swe-historical", 
                                                                cache_dir='/leonardo/home/userexternal/elenas00/projects/huggingface_htr/models/hub')

cer_metric = evaluate.load("cer")