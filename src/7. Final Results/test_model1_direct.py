#!/usr/bin/env python3
"""
Direct test of Model 1 to verify it's working correctly
"""
import sys
from pathlib import Path

# Add Model1 to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root / 'Model1_EmailContent'))

from phishing_detection_system import PhishingDetector
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_model1():
    """Test Model 1 directly"""
    model_path = project_root / 'trained_models' / 'Model1'
    
    if not model_path.exists():
        logger.error(f"Model not found at {model_path}")
        return False
    
    logger.info(f"Loading model from {model_path}")
    
    try:
        detector = PhishingDetector(require_cuda=False)
        detector.load_model(str(model_path.resolve()))
        
        logger.info("Model loaded successfully")
        logger.info(f"Model type: {type(detector.model)}")
        logger.info(f"Model is None: {detector.model is None}")
        
        # Test with a clear phishing email
        phishing_text = """
        Subject: URGENT: Your account has been suspended!
        
        Dear Customer,
        
        We have detected suspicious activity on your account. Your account has been temporarily suspended for security reasons.
        
        To verify your identity and restore access, please click the link below immediately:
        http://fake-bank-verify.com/login?token=abc123
        
        If you do not verify within 24 hours, your account will be permanently closed.
        
        This is an automated message. Please do not reply.
        """
        
        logger.info("\n=== Testing with PHISHING email ===")
        logger.info(f"Text length: {len(phishing_text)}")
        result = detector.predict(phishing_text)
        logger.info(f"Result: {result}")
        logger.info(f"Phishing prob: {result['phishing_probability']:.4f} ({result['phishing_probability']*100:.2f}%)")
        logger.info(f"Legitimate prob: {result['legitimate_probability']:.4f} ({result['legitimate_probability']*100:.2f}%)")
        logger.info(f"Is phishing: {result['is_phishing']}")
        
        if result['phishing_probability'] < 0.1:
            logger.error("ERROR: Model returned very low phishing probability for a clear phishing email!")
            logger.error("This suggests the model may not be working correctly.")
            return False
        
        # Test with a clear legitimate email
        legitimate_text = """
        Subject: Meeting Reminder
        
        Hi team,
        
        Just a reminder about our meeting tomorrow at 10 AM in the conference room.
        
        Please bring your project updates.
        
        Thanks,
        John
        """
        
        logger.info("\n=== Testing with LEGITIMATE email ===")
        logger.info(f"Text length: {len(legitimate_text)}")
        result2 = detector.predict(legitimate_text)
        logger.info(f"Result: {result2}")
        logger.info(f"Phishing prob: {result2['phishing_probability']:.4f} ({result2['phishing_probability']*100:.2f}%)")
        logger.info(f"Legitimate prob: {result2['legitimate_probability']:.4f} ({result2['legitimate_probability']*100:.2f}%)")
        logger.info(f"Is phishing: {result2['is_phishing']}")
        
        if result2['legitimate_probability'] < 0.5:
            logger.warning("WARNING: Model returned low legitimate probability for a clear legitimate email")
        
        logger.info("\n=== Model 1 Test Complete ===")
        return True
        
    except Exception as e:
        logger.error(f"ERROR: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False

if __name__ == "__main__":
    success = test_model1()
    sys.exit(0 if success else 1)

