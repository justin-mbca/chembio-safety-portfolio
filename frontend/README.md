# ChemBio SafeGuard Web Interface

A modern, responsive web interface for the Chemical & Biological Safety Assessment System.

## Features

- **Real-time Risk Assessment** - Instant safety analysis of chemical/biological content
- **Interactive Dashboard** - Visual risk indicators and confidence metrics
- **Responsive Design** - Works on desktop, tablet, and mobile devices
- **API Integration** - Seamless connection to the FastAPI backend
- **Example Queries** - Pre-built examples for common use cases
- **Live System Stats** - Real-time performance and health monitoring

## Quick Start

### Option 1: Complete System (Recommended)
```bash
# Start both API and frontend together
./start_system.sh
```

### Option 2: Frontend Only
```bash
# Start just the frontend (requires API running separately)
python frontend_server.py

# Or with custom port
python frontend_server.py --port 3001
```

## Usage

1. **Start the System**
   ```bash
   ./start_system.sh
   ```
   This will:
   - Activate the Python virtual environment
   - Install/update dependencies
   - Start the API server on port 8000
   - Start the frontend server on port 3000
   - Open your browser automatically

2. **Access the Interface**
   - Web Interface: http://localhost:3000
   - API Documentation: http://localhost:8000/docs
   - Health Check: http://localhost:8000/health

3. **Perform Safety Assessment**
   - Enter text related to chemistry, biology, or research
   - Optionally provide user ID and institution
   - Click "Analyze Safety Risk"
   - Review the comprehensive assessment results

## Interface Components

### Risk Assessment Panel
- **Text Input** - Enter content for safety analysis
- **Metadata Fields** - Optional user ID and institution
- **Example Queries** - Quick-start buttons with sample content
- **Results Display** - Comprehensive risk breakdown

### System Stats Panel
- **Model Parameters** - AI model specifications
- **Performance Metrics** - Latency and throughput statistics
- **Risk Categories** - Color-coded safety levels
- **API Status** - Real-time connection monitoring

## Risk Categories

| Category | Color | Description |
|----------|-------|-------------|
| **Benign** | Green | Safe educational content |
| **Low Risk** | Yellow | Educational with minor caution |
| **Medium Risk** | Orange | Requires safety awareness |
| **High Risk** | Red | Potentially harmful content |
| **Critical Risk** | Dark Red | Dangerous content requiring intervention |

## Assessment Results

Each analysis provides:

- **Risk Category** - Color-coded safety level
- **Risk Score** - Numerical confidence percentage
- **Confidence Level** - Model certainty in assessment
- **Processing Time** - Server and client latency
- **Request ID** - Unique identifier for tracking
- **Mitigation Action** - Recommended safety response
- **Detailed Explanation** - AI reasoning for the assessment

## Technical Details

### Frontend Stack
- **HTML5/CSS3** - Modern responsive layout
- **Vanilla JavaScript** - No framework dependencies
- **Font Awesome** - Professional icons
- **Fetch API** - RESTful backend communication

### Server Requirements
- **Python 3.8+** - Core runtime
- **Backend API** - ChemBio SafeGuard API service
- **Network Access** - Ports 8000 (API) and 3000 (Frontend)

### Browser Compatibility
- Chrome 88+
- Firefox 85+
- Safari 14+
- Edge 88+

## Development

### Customization
The interface can be customized by editing:
- `frontend/index.html` - Structure and styling
- `frontend_server.py` - Server configuration
- `start_system.sh` - Startup parameters

### API Configuration
Update the API base URL in the JavaScript:
```javascript
this.apiBaseUrl = 'http://localhost:8000';  // Change as needed
```

### Styling
The interface uses CSS Grid and Flexbox for responsive layout:
- Mobile-first responsive design
- Dark/light theme ready
- Custom risk category colors
- Smooth animations and transitions

## Security Notes

- The frontend communicates with localhost API only
- No data persistence in the web interface
- All assessments are processed server-side
- CORS headers configured for local development

## Troubleshooting

### Common Issues

**API Connection Failed**
- Ensure the API server is running on port 8000
- Check firewall settings
- Verify network connectivity

**Frontend Not Loading**
- Confirm frontend server is running on port 3000
- Check browser console for errors
- Try refreshing the page

**Port Already in Use**
```bash
# Use different ports
python frontend_server.py --port 3001
```

### Debug Mode
Enable browser developer tools (F12) to see:
- Network requests to the API
- Console logs and errors
- Performance timing information

## Support

For technical support or questions:
- Check the API documentation at `/docs`
- Review server logs for error details
- Submit issues to the GitHub repository

---

**ChemBio SafeGuard** - AI-Powered Chemical & Biological Safety Assessment System
