# 🚀 FREE Railway.app Deployment Guide

## Quick Start (5 minutes to live deployment!)

### **Option 1: Web Dashboard (Easiest)**

1. **Sign Up**: Go to [railway.app](https://railway.app) 
   - Click "Start a New Project"
   - Sign in with GitHub

2. **Deploy from GitHub**:
   - Select "Deploy from GitHub repo"
   - Choose `justin-mbca/chembio-safety-portfolio`
   - Railway auto-detects the setup ✨

3. **Configure Environment** (Optional):
   ```
   ENV=production
   LOG_LEVEL=INFO
   ```

4. **Deploy**: Click "Deploy" - takes ~3-5 minutes
   
5. **Access**: Get your free URL like:
   ```
   https://chembio-safety-production.railway.app
   ```

### **Option 2: Railway CLI (For Developers)**

```bash
# Install Railway CLI
npm install -g @railway/cli

# Login with GitHub
railway login

# Deploy from current directory
cd /Users/justin/chembio-safety-portfolio
railway deploy

# Get deployment URL
railway open
```

---

## 🎯 **What You Get FREE on Railway**

✅ **$5 monthly credit** (auto-renews, no card required)  
✅ **Custom subdomain**: `your-app.railway.app`  
✅ **Automatic HTTPS**: SSL certificate included  
✅ **Auto-deployments**: Updates on git push  
✅ **Environment variables**: Easy configuration  
✅ **Monitoring**: Built-in metrics and logs  
✅ **Zero configuration**: Detects Python/Docker automatically  

---

## 🔧 **Railway Configuration Files**

### **railway.json** (Already created ✅)
```json
{
  "$schema": "https://railway.app/railway.schema.json",
  "build": {
    "builder": "DOCKERFILE"
  },
  "deploy": {
    "startCommand": "python run.py",
    "healthcheckPath": "/health",
    "restartPolicyType": "ON_FAILURE"
  }
}
```

### **Environment Variables (Set in Railway Dashboard)**
```
PORT=3000                    # Railway sets this automatically
ENV=production
LOG_LEVEL=INFO
REDIS_URL=                   # Leave empty for now (can add Redis later)
```

---

## 📱 **After Deployment**

### **1. Test Your Deployment**
```bash
# Test API health
curl https://your-app.railway.app/health

# Test safety assessment
curl -X POST https://your-app.railway.app/assess-safety \
  -H "Content-Type: application/json" \
  -d '{"text": "What is water?", "return_explanation": true}'

# Test enhanced assessment (Phase 1)
curl -X POST https://your-app.railway.app/assess-safety-enhanced \
  -H "Content-Type: application/json" \
  -d '{"text": "laboratory safety protocols", "return_explanation": true}'
```

### **2. Access Your App**
- **🌐 Main App**: `https://your-app.railway.app`
- **📚 API Docs**: `https://your-app.railway.app/docs`
- **❤️ Health Check**: `https://your-app.railway.app/health`

### **3. Monitor Your App**
- Railway dashboard shows real-time metrics
- View logs directly in the web interface
- Monitor usage against your $5 monthly credit

---

## 🔄 **Automatic Deployments**

Railway automatically redeploys when you push to GitHub:

```bash
# Make changes locally
git add .
git commit -m "Update feature"
git push origin main

# Railway automatically deploys the changes! 🚀
```

---

## 💡 **Pro Tips**

### **Add Redis Later (If Needed)**
```bash
# In Railway dashboard:
# 1. Add "Redis" service to your project
# 2. Railway provides REDIS_URL automatically
# 3. Your app connects automatically!
```

### **Custom Domain (Free)**
```bash
# In Railway dashboard:
# 1. Go to Settings > Domains
# 2. Add your custom domain
# 3. Update DNS records
# 4. Get free SSL certificate
```

### **Monitor Usage**
```bash
# Railway dashboard shows:
# - Memory usage
# - CPU usage  
# - Request count
# - Monthly credit usage
```

---

## 🎯 **Expected Results**

After deployment, your ChemBio SafeGuard will be:

✅ **Live on the internet** with HTTPS  
✅ **Accessible via API** for integrations  
✅ **Auto-scaling** based on traffic  
✅ **Monitoring enabled** with logs  
✅ **Phase 1 database integration** working  
✅ **Zero monthly cost** (under $5 usage)  

---

**Ready to deploy? Go to [railway.app](https://railway.app) and connect your GitHub repo!**

The deployment will be live in ~5 minutes. 🚀
