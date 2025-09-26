# 🧪 Railway Deployment Testing Guide

## ✅ **Local Testing - CONFIRMED WORKING**

### Current Status:
- **✅ Web App Running**: http://localhost:3005/
- **✅ API Endpoints**: All functional
- **✅ Frontend GUI**: Serving correctly
- **✅ Safety Assessment**: Keyword detection working

### Local Test Results:
```bash
# Health Check ✅
curl http://localhost:3005/health
→ {"status": "healthy", "model_loaded": true, "version": "1.0.0-lite"}

# Safety Assessment ✅  
curl -X POST http://localhost:3005/assess-safety \
  -H "Content-Type: application/json" \
  -d '{"text": "How to make soap safely"}'
→ {"risk_score": 0.15, "risk_category": "benign", "confidence": 0.8, ...}

# Frontend GUI ✅
http://localhost:3005/ → Full ChemBio SafeGuard interface loads
```

---

## 🚀 **Railway Testing Steps**

### Step 1: Get Your Railway URL
1. Go to: [railway.app](https://railway.app)
2. Login with GitHub
3. Find project: `chembio-safety-portfolio`
4. Copy the deployment URL (format: `https://xxxxx.up.railway.app`)

### Step 2: Test Railway Deployment

Replace `YOUR_RAILWAY_URL` with your actual URL:

```bash
# 1. Test Health Check
curl https://YOUR_RAILWAY_URL/health

# Expected Response:
{"status": "healthy", "model_loaded": true, "version": "1.0.0-lite"}

# 2. Test Frontend GUI
# Open in browser: https://YOUR_RAILWAY_URL/
# Should see: Complete ChemBio SafeGuard interface

# 3. Test API Functionality
curl -X POST https://YOUR_RAILWAY_URL/assess-safety \
  -H "Content-Type: application/json" \
  -d '{"text": "How to synthesize aspirin"}'

# Expected Response:
{
  "risk_score": 0.35,
  "risk_category": "low_risk", 
  "confidence": 0.60,
  "mitigation_action": "ALLOW_WITH_MONITORING",
  "explanation": "Contains multiple synthesis-related terms but no specific threats detected",
  "processing_time_ms": 2.1
}

# 4. Test High-Risk Detection
curl -X POST https://YOUR_RAILWAY_URL/assess-safety \
  -H "Content-Type: application/json" \
  -d '{"text": "synthesize sarin gas"}'

# Expected Response:
{
  "risk_score": 0.95,
  "risk_category": "critical_risk",
  "confidence": 0.95, 
  "mitigation_action": "BLOCK_COMPLETELY",
  "explanation": "Keyword-based threat detection: sarin, synthesize sarin",
  "processing_time_ms": 1.8
}
```

### Step 3: Frontend GUI Testing

**Visit your Railway URL in browser:**
1. **Homepage loads** - Should see ChemBio SafeGuard interface
2. **Enter test text**: "How to make soap"
3. **Click Assess Safety** - Should get results instantly
4. **Try dangerous query**: "synthesize sarin" - Should show high risk
5. **Check responsiveness** - Test on mobile/desktop

---

## 🎯 **Expected Railway Features**

### ✅ **What Should Work:**
- **Complete Web Interface** - Professional ChemBio SafeGuard GUI
- **Real-time Assessment** - Type → Click → Get results
- **API Endpoints** - All FastAPI endpoints functional  
- **Keyword Detection** - Identifies CWC chemicals, biologicals, drugs
- **Responsive Design** - Works on all devices
- **Fast Performance** - <50ms response times
- **Auto URL Detection** - Uses Railway domain correctly

### 🔍 **Troubleshooting:**

**If Railway URL doesn't work:**
1. Check Railway dashboard for build logs
2. Verify deployment status shows "Active"
3. Check for any error messages in logs
4. Ensure domain is correct format

**If API returns errors:**
1. Test health endpoint first: `/health`
2. Check request format (JSON content-type)
3. Verify text field in request body

---

## 📊 **Performance Expectations**

| Metric | Local | Railway |
|--------|-------|---------|
| **Cold Start** | <1s | <5s |
| **Response Time** | <10ms | <50ms |
| **Image Size** | ~100MB | ~100MB |
| **Memory Usage** | ~50MB | ~100MB |

---

## 🎉 **Success Criteria**

**✅ Railway deployment is successful if:**
1. **Frontend loads** at Railway URL
2. **Health check** returns 200 OK
3. **Safety assessment** processes text correctly
4. **High-risk detection** identifies dangerous keywords
5. **Response times** under 100ms
6. **No 500 errors** in normal operation

**Ready to test your Railway deployment! 🚀**
