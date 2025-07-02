import { GeneratedFile } from './project-generator';

export class ZipGenerator {
  static async createProjectZip(files: GeneratedFile[], projectName: string): Promise<Blob> {
    // Since we can't use node libraries in the browser, we'll create a simple ZIP-like structure
    // In a real implementation, you'd use a library like JSZip
    
    const zipContent = this.createZipContent(files, projectName);
    return new Blob([zipContent], { type: 'application/zip' });
  }

  private static createZipContent(files: GeneratedFile[], projectName: string): string {
    // Create a simple text representation of the project structure
    // In production, this would be actual ZIP binary data
    
    let content = `# ${projectName} Project Files\n\n`;
    content += `Generated on: ${new Date().toISOString()}\n`;
    content += `Total files: ${files.length}\n\n`;
    content += '=' .repeat(50) + '\n\n';

    files.forEach(file => {
      content += `## File: ${file.path}\n`;
      content += '-'.repeat(30) + '\n';
      content += file.content;
      content += '\n\n' + '='.repeat(50) + '\n\n';
    });

    return content;
  }

  static downloadFile(blob: Blob, filename: string): void {
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.download = filename;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    URL.revokeObjectURL(url);
  }
}