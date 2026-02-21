---
description: Build, commit, push, and verify Vercel deployment
---

// turbo-all

1. Run the full production build to catch any errors:
```
npm run build
```

2. If build passes, stage all changes:
```
git add -A
```

3. Commit with a descriptive message:
```
git commit -m "<message>"
```

4. Push to remote:
```
git push
```

5. Wait 90 seconds, then check Vercel deployment status:
```powershell
Start-Sleep -Seconds 90; $resp = Invoke-RestMethod -Uri "https://api.vercel.com/v6/deployments?limit=1" -Headers @{Authorization="Bearer $env:VERCEL_TOKEN"}; $d = $resp.deployments[0]; Write-Output "State: $($d.state) | URL: $($d.url)"
```

6. If deployment state is ERROR, check the error:
```powershell
$resp = Invoke-RestMethod -Uri "https://api.vercel.com/v6/deployments?limit=1" -Headers @{Authorization="Bearer $env:VERCEL_TOKEN"}; $uid = $resp.deployments[0].uid; $detail = Invoke-RestMethod -Uri "https://api.vercel.com/v13/deployments/$uid" -Headers @{Authorization="Bearer $env:VERCEL_TOKEN"}; Write-Output "Error: $($detail.errorMessage)"; Write-Output "Code: $($detail.errorCode)"
```

**Note**: The VERCEL_TOKEN environment variable must be set before running steps 5-6:
```powershell
$env:VERCEL_TOKEN = "<token>"
```
