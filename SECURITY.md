# Security Policy

## Supported Versions

Currently, only the latest version of the Learning Analytics Framework is supported with security updates.

| Version | Supported |
|---------|------------|
| Latest   | ✅        |

## Reporting a Vulnerability

If you discover a security vulnerability, please report it responsibly.

### How to Report

**Do NOT** open a public issue for security vulnerabilities.

Instead, please send an email to: [your-security-email@example.com]

Include the following information:

- Description of the vulnerability
- Steps to reproduce the vulnerability
- Potential impact of the vulnerability
- Suggested fix (if known)

### What to Expect

- We will acknowledge receipt of your report within 48 hours
- We will provide a detailed response within 7 days
- We will work with you to understand and fix the vulnerability
- We will coordinate a release schedule for the fix
- We will credit you in the release notes (unless you prefer anonymity)

## Security Best Practices

### For Users

1. **Keep Dependencies Updated**: Regularly update Python packages
   ```bash
   pip install --upgrade -r requirements.txt
   ```

2. **Review Input Data**: Validate that input Excel files come from trusted sources
   - The pipeline reads data from Excel files which could contain malicious content
   - Only process data from trusted educational institutions or sources

3. **Environment Isolation**: Use virtual environments to isolate dependencies
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

4. **Access Control**: Restrict access to sensitive student data
   - Student data may contain personally identifiable information (PII)
   - Follow your institution's data governance policies
   - Consider encryption for data at rest and in transit

5. **File Permissions**: Ensure appropriate file permissions on input/output directories
   - Restrict write access to output directories
   - Secure log files that may contain student information

### For Developers

1. **Input Validation**: The pipeline includes data validation (Phase 1)
   - Review and extend validation rules for your specific use case
   - Never trust user input without validation

2. **Dependency Management**: 
   - Review security advisories for dependencies
   - Use `pip-audit` to check for known vulnerabilities
   ```bash
   pip install pip-audit
   pip-audit
   ```

3. **Secrets Management**: 
   - Never commit API keys, passwords, or credentials
   - Use environment variables for sensitive configuration
   - The `.gitignore` file prevents accidental commits of sensitive files

4. **Code Review**: All code changes should be reviewed for security implications
   - Check for hardcoded credentials
   - Validate file handling operations
   - Review data processing logic for injection vulnerabilities

## Known Security Considerations

### Data Privacy

- **Student Data**: This framework processes student assessment data which may be sensitive
  - Follow FERPA, GDPR, or other applicable regulations
  - Implement appropriate data retention policies
  - Consider anonymization for research publications

- **Model Artifacts**: Trained models may memorize patterns from training data
  - Be cautious about sharing model artifacts publicly
  - Consider differential privacy techniques if needed

### File Handling

- **Excel Files**: The pipeline processes Excel files which can contain macros
  - Input files should come from trusted sources
  - Consider using `.xlsx` format (no macros) instead of `.xlsm`

- **Path Traversal**: The framework uses pathlib for safe path handling
  - All file operations use absolute paths from the project root
  - No user-supplied paths are used without validation

### External Dependencies

- **Optional Dependencies**: `xgboost` and `shap` are optional
  - Review these packages before installation if security is critical
  - The pipeline works without them using built-in alternatives

## Security-Related Features

### Built-in Protections

1. **Read-Only Validation**: Phase 1 validates data without modifying source files
2. **Path Isolation**: All operations are contained within the project directory
3. **Logging**: Comprehensive logging for audit trails
4. **Error Handling**: Defensive error handling prevents information leakage

### Recommendations

1. **Input Sanitization**: Validate all input files before processing
2. **Output Encryption**: Consider encrypting output files containing sensitive data
3. **Access Logging**: Monitor who accesses the pipeline and its outputs
4. **Regular Audits**: Periodically review access logs and data handling procedures

## Contact

For security-related questions or concerns:
- **Email**: [your-security-email@example.com]
- **GitHub Issues**: For non-sensitive security discussions

---

Thank you for helping keep the Learning Analytics Framework secure! 🔒
