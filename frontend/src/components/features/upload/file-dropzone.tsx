"use client";

import { useCallback, useState } from "react";
import { useDropzone } from "react-dropzone";
import { Upload, File, X, AlertCircle, CheckCircle } from "lucide-react";
import { Button } from "@/components/ui/button";
import { cn, formatFileSize } from "@/lib/utils";

interface FileDropzoneProps {
  onFileSelect: (file: File) => void;
  accept?: Record<string, string[]>;
  maxSize?: number;
  disabled?: boolean;
}

export function FileDropzone({
  onFileSelect,
  accept = {
    "text/plain": [".fasta", ".fa", ".fna", ".fastq", ".fq"],
    "application/gzip": [".gz"],
  },
  maxSize = 100 * 1024 * 1024, // 100MB
  disabled = false,
}: FileDropzoneProps) {
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [error, setError] = useState<string | null>(null);

  const onDrop = useCallback(
    (acceptedFiles: File[], fileRejections: { file: File; errors: readonly { message: string; code: string }[] }[]) => {
      setError(null);

      if (fileRejections.length > 0) {
        const rejection = fileRejections[0];
        if (rejection.errors[0]) {
          setError(rejection.errors[0].message);
        }
        return;
      }

      if (acceptedFiles.length > 0) {
        const file = acceptedFiles[0];
        setSelectedFile(file);
        onFileSelect(file);
      }
    },
    [onFileSelect]
  );

  const { getRootProps, getInputProps, isDragActive, isDragReject } = useDropzone({
    onDrop,
    accept,
    maxSize,
    maxFiles: 1,
    disabled,
  });

  const removeFile = () => {
    setSelectedFile(null);
    setError(null);
  };

  const isValidFile = (filename: string) => {
    const validExtensions = [".fasta", ".fa", ".fna", ".fastq", ".fq", ".gz"];
    return validExtensions.some((ext) => filename.toLowerCase().endsWith(ext));
  };

  return (
    <div className="space-y-4">
      <div
        {...getRootProps()}
        className={cn(
          "relative border-2 border-dashed rounded-xl p-8 text-center cursor-pointer transition-all duration-200",
          isDragActive && !isDragReject && "border-primary bg-primary/5",
          isDragReject && "border-destructive bg-destructive/5",
          !isDragActive && !error && "border-border hover:border-primary/50 hover:bg-surface-hover",
          error && "border-destructive",
          disabled && "opacity-50 cursor-not-allowed"
        )}
      >
        <input {...getInputProps()} />

        {selectedFile ? (
          <div className="flex items-center justify-center gap-4">
            <div className="flex items-center gap-3 px-4 py-3 rounded-lg bg-primary/10 border border-primary/20">
              <File className="h-5 w-5 text-primary" />
              <div className="text-left">
                <p className="text-sm font-medium text-foreground">{selectedFile.name}</p>
                <p className="text-xs text-muted">{formatFileSize(selectedFile.size)}</p>
              </div>
              <Button
                variant="ghost"
                size="icon"
                className="h-6 w-6 ml-2"
                onClick={(e) => {
                  e.stopPropagation();
                  removeFile();
                }}
              >
                <X className="h-4 w-4" />
              </Button>
            </div>
          </div>
        ) : (
          <>
            <div className="w-16 h-16 mx-auto mb-4 rounded-full bg-primary/10 flex items-center justify-center">
              <Upload className={cn("h-8 w-8", isDragActive ? "text-primary" : "text-muted")} />
            </div>
            <p className="text-foreground font-medium mb-2">
              {isDragActive ? "Drop your file here" : "Drag & drop your genomic sequence file"}
            </p>
            <p className="text-sm text-muted mb-4">
              or click to browse files
            </p>
            <div className="flex items-center justify-center gap-2 text-xs text-muted">
              <span className="px-2 py-1 rounded bg-surface-hover">.fasta</span>
              <span className="px-2 py-1 rounded bg-surface-hover">.fastq</span>
              <span className="px-2 py-1 rounded bg-surface-hover">.fa</span>
              <span className="px-2 py-1 rounded bg-surface-hover">.fq</span>
              <span className="px-2 py-1 rounded bg-surface-hover">.gz</span>
            </div>
            <p className="text-xs text-muted mt-4">Maximum file size: {formatFileSize(maxSize)}</p>
          </>
        )}
      </div>

      {error && (
        <div className="flex items-center gap-2 p-3 rounded-lg bg-destructive/10 border border-destructive/20 text-destructive">
          <AlertCircle className="h-4 w-4 flex-shrink-0" />
          <span className="text-sm">{error}</span>
        </div>
      )}

      {selectedFile && isValidFile(selectedFile.name) && !error && (
        <div className="flex items-center gap-2 p-3 rounded-lg bg-susceptible/10 border border-susceptible/20 text-susceptible">
          <CheckCircle className="h-4 w-4 flex-shrink-0" />
          <span className="text-sm">File format validated successfully</span>
        </div>
      )}
    </div>
  );
}
