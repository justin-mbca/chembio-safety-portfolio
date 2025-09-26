# Ultra-minimal ChemBio SafeGuard for Railway deployment
# This is a lightweight API-only version

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
import time
import logging
from typing import Optional

# Simple keyword-based safety checker (no ML model to reduce size)
DANGEROUS_KEYWORDS = {
    # Chemical weapons (CWC Schedule 1)
    'sarin', 'tabun', 'soman', 'vx', 'mustard gas', 'lewisite', 'ricin',
    # Biological agents  
    'anthrax', 'botulinum', 'ebola', 'plague', 'smallpox', 'tularemia',
    # Drug precursors
    'pseudoephedrine', 'ephedrine', 'methylamine', 'phenylacetic acid',
    # Explosives
    'ammonium nitrate', 'potassium nitrate', 'hydrogen peroxide', 'acetone peroxide',
    # Synthesis keywords
    'synthesize sarin', 'make explosives', 'biological weapon', 'chemical weapon'
}

app = FastAPI(title="ChemBio SafeGuard Lite", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class SafetyRequest(BaseModel):
    text: str
    return_explanation: Optional[bool] = False

class SafetyResponse(BaseModel):
    risk_score: float
    risk_category: str
    confidence: float
    mitigation_action: str
    explanation: str
    processing_time_ms: float

@app.get("/health")
def health_check():
    return {"status": "healthy", "model_loaded": True, "version": "1.0.0-lite"}

@app.post("/assess-safety", response_model=SafetyResponse)
@app.post("/assess-safety-enhanced", response_model=SafetyResponse)
def assess_safety(request: SafetyRequest):
    start_time = time.time()
    
    text_lower = request.text.lower()
    
    # Check for dangerous keywords
    dangerous_matches = []
    for keyword in DANGEROUS_KEYWORDS:
        if keyword in text_lower:
            dangerous_matches.append(keyword)
    
    # Calculate risk based on keyword matches
    if dangerous_matches:
        risk_score = 0.95  # High risk for keyword matches
        risk_category = "critical_risk"
        confidence = 0.95
        action = "BLOCK_COMPLETELY"
        explanation = f"Keyword-based threat detection: {', '.join(dangerous_matches)}"
    else:
        # Simple length and complexity heuristics
        suspicious_terms = ['synthesize', 'create', 'make', 'produce', 'manufacture']
        suspicious_count = sum(1 for term in suspicious_terms if term in text_lower)
        
        if suspicious_count >= 2:
            risk_score = 0.35
            risk_category = "low_risk"
            confidence = 0.60
            action = "ALLOW_WITH_MONITORING"
            explanation = "Contains multiple synthesis-related terms but no specific threats detected"
        else:
            risk_score = 0.15
            risk_category = "benign"
            confidence = 0.80
            action = "ALLOW_WITH_MONITORING"
            explanation = "No specific safety concerns detected"
    
    processing_time = (time.time() - start_time) * 1000
    
    return SafetyResponse(
        risk_score=risk_score,
        risk_category=risk_category,
        confidence=confidence,
        mitigation_action=action,
        explanation=explanation,
        processing_time_ms=processing_time
    )

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 3000))
    uvicorn.run("simple_api_lite:app", host="0.0.0.0", port=port)
