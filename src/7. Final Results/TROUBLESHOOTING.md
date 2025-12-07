# Troubleshooting Guide

## Model Loading Issues

### Problem: "The paging file is too small" Error

**Symptoms:**
- Model 1 (Email Content) fails to load
- Error message mentions "paging file" or "virtual memory"
- Server starts but predictions fail

**Cause:**
Windows virtual memory (paging file) is too small to load the machine learning models.

**Solution:**

#### Option 1: Increase Paging File (Recommended)

1. **Run the helper script:**
   ```powershell
   cd api_server
   .\fix_paging_file.ps1
   ```

2. **Or manually:**
   - Press `Win + R`, type `sysdm.cpl`, press Enter
   - Go to "Advanced" tab → "Performance" → "Settings"
   - Go to "Advanced" tab → "Virtual memory" → "Change"
   - Uncheck "Automatically manage paging file size"
   - Select your system drive → "Custom size"
   - Set Initial: `8192` MB, Maximum: `16384` MB
   - Click "Set" → "OK" → **Restart computer**

#### Option 2: Use Only Working Models

The server will work with whichever models load successfully:
- If Model 1 fails, you can still use Model 2 (URL detection)
- If Model 2 fails, you can still use Model 1 (Email content)
- Check `/health` endpoint to see which models are loaded

#### Option 3: Close Other Applications

Free up memory by closing:
- Web browsers with many tabs
- Video editors
- Other memory-intensive applications

### Problem: "Python was not found"

**Solution:**
Use the full path to Python:
```powershell
N:\python\python.exe app.py
```

Or add Python to your PATH environment variable.

### Problem: Models Load Slowly

**This is normal!** Models load on-demand (lazy loading):
- First prediction request will take 10-30 seconds (model loading)
- Subsequent requests are fast (< 1 second)

### Problem: Server Won't Start

**Check:**
1. Port 5000 is not in use:
   ```powershell
   netstat -ano | findstr :5000
   ```
2. Python is installed correctly
3. Dependencies are installed:
   ```powershell
   pip install -r api_server/requirements.txt
   ```

## Testing

### Test Model 2 (URL Detection):
```powershell
curl -X POST http://localhost:5000/predict-url -H "Content-Type: application/json" -d '{\"url\": \"https://example.com\"}'
```

### Test Model 1 (Email Content):
```powershell
curl -X POST http://localhost:5000/predict-email -H "Content-Type: application/json" -d '{\"email_text\": \"Subject: Test\nBody: Test email\"}'
```

### Check Server Status:
```powershell
curl http://localhost:5000/health
```

## Getting Help

If issues persist:
1. Check server logs for detailed error messages
2. Verify models are trained and saved in `trained_models/` folder
3. Check available memory: `Get-CimInstance Win32_OperatingSystem | Select-Object TotalVirtualMemorySize, FreeVirtualMemory`
4. See `MEMORY_OPTIMIZATION.md` for more details




