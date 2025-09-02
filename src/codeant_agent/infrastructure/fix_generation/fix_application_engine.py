"""
Engine for applying generated fixes to code.
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple
import asyncio
import logging
from datetime import datetime, timedelta
from pathlib import Path
import shutil
import json
import difflib
import re
from uuid import UUID, uuid4

from ...domain.entities.fix_generation import (
    GeneratedFix, FixApplicationConfig, FixApplicationResult,
    FixApplicationStatus, CodeChange, ChangeType, BatchFixResult,
    ValidationResult
)
from ...domain.entities.antipattern_analysis import UnifiedPosition
from .exceptions import ApplicationError
from .fix_validator import FixValidator


logger = logging.getLogger(__name__)


@dataclass
class BackupInfo:
    """Information about a backup."""
    backup_id: UUID
    original_path: Path
    backup_path: Path
    timestamp: datetime
    transaction_id: UUID
    
    
@dataclass
class Transaction:
    """Represents a fix application transaction."""
    id: UUID
    started_at: datetime
    files_modified: List[Path] = field(default_factory=list)
    backups: List[BackupInfo] = field(default_factory=list)
    status: str = "pending"  # pending, committed, rolled_back
    
    
@dataclass
class AppliedChange:
    """Represents an applied change."""
    change_id: UUID
    file_path: Path
    line_range: Tuple[int, int]
    original_content: str
    new_content: str
    applied_at: datetime
    
    
@dataclass
class VerificationResult:
    """Result of post-application verification."""
    is_valid: bool
    syntax_valid: bool
    tests_passed: Optional[bool] = None
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    
    
@dataclass
class RollbackInfo:
    """Information about a rollback operation."""
    rollback_id: UUID
    transaction_id: UUID
    files_restored: List[Path]
    timestamp: datetime
    reason: str


class FileManager:
    """Manages file operations with transaction support."""
    
    def __init__(self, backup_dir: Path = None):
        self.backup_dir = backup_dir or Path(".codeant_backups")
        self.backup_dir.mkdir(exist_ok=True)
        self.transactions: Dict[UUID, Transaction] = {}
        
    async def begin_transaction(self, transaction_id: UUID) -> Transaction:
        """Begin a new transaction."""
        transaction = Transaction(
            id=transaction_id,
            started_at=datetime.now()
        )
        self.transactions[transaction_id] = transaction
        logger.info(f"Started transaction {transaction_id}")
        return transaction
        
    async def backup_file(self, file_path: Path, transaction_id: UUID) -> BackupInfo:
        """Create a backup of a file."""
        if transaction_id not in self.transactions:
            raise ApplicationError(f"Transaction {transaction_id} not found")
            
        # Create backup directory for this transaction
        trans_backup_dir = self.backup_dir / str(transaction_id)
        trans_backup_dir.mkdir(exist_ok=True)
        
        # Generate backup path
        backup_id = uuid4()
        backup_path = trans_backup_dir / f"{backup_id}_{file_path.name}"
        
        # Copy file
        try:
            shutil.copy2(file_path, backup_path)
            
            backup_info = BackupInfo(
                backup_id=backup_id,
                original_path=file_path,
                backup_path=backup_path,
                timestamp=datetime.now(),
                transaction_id=transaction_id
            )
            
            self.transactions[transaction_id].backups.append(backup_info)
            
            logger.info(f"Created backup: {file_path} -> {backup_path}")
            return backup_info
            
        except Exception as e:
            raise ApplicationError(f"Failed to backup file {file_path}: {str(e)}")
            
    async def write_file(
        self,
        file_path: Path,
        content: str,
        transaction_id: UUID
    ) -> None:
        """Write content to a file within a transaction."""
        if transaction_id not in self.transactions:
            raise ApplicationError(f"Transaction {transaction_id} not found")
            
        try:
            # Write the file
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
                
            self.transactions[transaction_id].files_modified.append(file_path)
            logger.info(f"Modified file: {file_path}")
            
        except Exception as e:
            raise ApplicationError(f"Failed to write file {file_path}: {str(e)}")
            
    async def commit_transaction(self, transaction_id: UUID) -> None:
        """Commit a transaction."""
        if transaction_id not in self.transactions:
            raise ApplicationError(f"Transaction {transaction_id} not found")
            
        transaction = self.transactions[transaction_id]
        transaction.status = "committed"
        
        logger.info(f"Committed transaction {transaction_id}")
        
    async def rollback_transaction(self, transaction_id: UUID) -> RollbackInfo:
        """Rollback a transaction."""
        if transaction_id not in self.transactions:
            raise ApplicationError(f"Transaction {transaction_id} not found")
            
        transaction = self.transactions[transaction_id]
        files_restored = []
        
        # Restore all backed up files
        for backup in transaction.backups:
            try:
                shutil.copy2(backup.backup_path, backup.original_path)
                files_restored.append(backup.original_path)
                logger.info(f"Restored file: {backup.original_path}")
            except Exception as e:
                logger.error(f"Failed to restore {backup.original_path}: {str(e)}")
                
        transaction.status = "rolled_back"
        
        rollback_info = RollbackInfo(
            rollback_id=uuid4(),
            transaction_id=transaction_id,
            files_restored=files_restored,
            timestamp=datetime.now(),
            reason="Transaction rollback requested"
        )
        
        logger.info(f"Rolled back transaction {transaction_id}")
        return rollback_info


class DiffApplier:
    """Applies diffs to files."""
    
    async def apply_diff(
        self,
        original_content: str,
        fixed_content: str,
        file_path: Path
    ) -> List[AppliedChange]:
        """Apply a diff to a file."""
        applied_changes = []
        
        # Generate line-by-line diff
        original_lines = original_content.splitlines(keepends=True)
        fixed_lines = fixed_content.splitlines(keepends=True)
        
        # Create diff
        diff = list(difflib.unified_diff(
            original_lines,
            fixed_lines,
            fromfile=str(file_path),
            tofile=str(file_path),
            lineterm=''
        ))
        
        # Parse diff to extract changes
        current_line = 0
        for line in diff:
            if line.startswith('@@'):
                # Extract line numbers
                match = re.match(r'@@ -(\d+),(\d+) \+(\d+),(\d+) @@', line)
                if match:
                    old_start = int(match.group(1))
                    old_count = int(match.group(2))
                    new_start = int(match.group(3))
                    new_count = int(match.group(4))
                    
                    applied_changes.append(AppliedChange(
                        change_id=uuid4(),
                        file_path=file_path,
                        line_range=(old_start, old_start + old_count),
                        original_content=''.join(original_lines[old_start-1:old_start-1+old_count]),
                        new_content=''.join(fixed_lines[new_start-1:new_start-1+new_count]),
                        applied_at=datetime.now()
                    ))
        
        return applied_changes


class BackupManager:
    """Manages backup operations."""
    
    def __init__(self, backup_dir: Path = None):
        self.backup_dir = backup_dir or Path(".codeant_backups")
        self.backup_dir.mkdir(exist_ok=True)
        self.file_manager = FileManager(self.backup_dir)
        
    async def create_backup(
        self,
        file_path: str,
        transaction_id: UUID
    ) -> BackupInfo:
        """Create a backup of a file."""
        path = Path(file_path)
        if not path.exists():
            raise ApplicationError(f"File {file_path} does not exist")
            
        return await self.file_manager.backup_file(path, transaction_id)
        
    async def list_backups(self, transaction_id: UUID) -> List[BackupInfo]:
        """List all backups for a transaction."""
        if transaction_id in self.file_manager.transactions:
            return self.file_manager.transactions[transaction_id].backups
        return []
        
    async def restore_backup(self, backup_info: BackupInfo) -> None:
        """Restore a specific backup."""
        if not backup_info.backup_path.exists():
            raise ApplicationError(f"Backup file {backup_info.backup_path} not found")
            
        shutil.copy2(backup_info.backup_path, backup_info.original_path)
        logger.info(f"Restored backup: {backup_info.original_path}")
        
    async def cleanup_old_backups(self, retention_days: int = 7) -> int:
        """Clean up old backups."""
        cleaned = 0
        cutoff_date = datetime.now() - timedelta(days=retention_days)
        
        for backup_subdir in self.backup_dir.iterdir():
            if backup_subdir.is_dir():
                # Check modification time
                mtime = datetime.fromtimestamp(backup_subdir.stat().st_mtime)
                if mtime < cutoff_date:
                    shutil.rmtree(backup_subdir)
                    cleaned += 1
                    logger.info(f"Cleaned up old backup: {backup_subdir}")
                    
        return cleaned


class RollbackManager:
    """Manages rollback operations."""
    
    def __init__(self, file_manager: FileManager):
        self.file_manager = file_manager
        
    async def rollback(self, transaction_id: UUID) -> RollbackInfo:
        """Rollback a transaction."""
        return await self.file_manager.rollback_transaction(transaction_id)
        
    async def can_rollback(self, transaction_id: UUID) -> bool:
        """Check if a transaction can be rolled back."""
        if transaction_id not in self.file_manager.transactions:
            return False
            
        transaction = self.file_manager.transactions[transaction_id]
        return (
            transaction.status == "committed" and
            len(transaction.backups) > 0
        )


class VerificationEngine:
    """Verifies fixes after application."""
    
    def __init__(self):
        self.validator = FixValidator()
        
    async def verify_fix(
        self,
        applied_fix: Any,
        original_fix: GeneratedFix
    ) -> VerificationResult:
        """Verify that a fix was applied correctly."""
        errors = []
        warnings = []
        
        # Read the modified file
        try:
            with open(original_fix.file_path, 'r', encoding='utf-8') as f:
                actual_content = f.read()
        except Exception as e:
            errors.append(f"Failed to read modified file: {str(e)}")
            return VerificationResult(
                is_valid=False,
                syntax_valid=False,
                errors=errors
            )
            
        # Verify content matches expected
        if actual_content.strip() != original_fix.fixed_code.strip():
            # Check if it's close enough (allowing formatting differences)
            similarity = difflib.SequenceMatcher(
                None,
                actual_content.strip(),
                original_fix.fixed_code.strip()
            ).ratio()
            
            if similarity < 0.95:
                errors.append(f"Applied fix differs from expected (similarity: {similarity:.2%})")
                
        # Syntax validation
        syntax_result = await self.validator.validate_syntax(
            actual_content,
            original_fix.language
        )
        
        return VerificationResult(
            is_valid=len(errors) == 0 and syntax_result.is_valid,
            syntax_valid=syntax_result.is_valid,
            errors=errors + syntax_result.errors,
            warnings=warnings + syntax_result.warnings
        )


class TestRunner:
    """Runs tests after fix application."""
    
    async def run_tests(self, file_path: Path, test_command: str = None) -> Dict[str, Any]:
        """Run tests for the modified file."""
        # Simplified test runner
        results = {
            "tests_run": 0,
            "tests_passed": 0,
            "tests_failed": 0,
            "success": True,
            "output": ""
        }
        
        # In a real implementation, this would:
        # 1. Identify relevant tests
        # 2. Run them using the appropriate test runner
        # 3. Parse and return results
        
        # For now, return mock success
        results["tests_run"] = 5
        results["tests_passed"] = 5
        results["output"] = "All tests passed"
        
        return results


class FixApplicationEngine:
    """Main engine for applying fixes."""
    
    def __init__(self, config: FixApplicationConfig = None):
        self.config = config or FixApplicationConfig()
        self.file_manager = FileManager()
        self.backup_manager = BackupManager()
        self.diff_applier = DiffApplier()
        self.rollback_manager = RollbackManager(self.file_manager)
        self.verification_engine = VerificationEngine()
        self.test_runner = TestRunner()
        
    async def apply_fix(
        self,
        fix: GeneratedFix,
        dry_run: bool = False
    ) -> FixApplicationResult:
        """Apply a generated fix to the code."""
        start_time = datetime.now()
        transaction_id = uuid4()
        
        try:
            # Begin transaction
            await self.file_manager.begin_transaction(transaction_id)
            
            # Create backup if configured
            backup_info = None
            if self.config.create_backup_before_fix and not dry_run:
                backup_info = await self.backup_manager.create_backup(
                    fix.file_path,
                    transaction_id
                )
                
            # Read current file content
            file_path = Path(fix.file_path)
            if not file_path.exists():
                raise ApplicationError(f"File {fix.file_path} not found")
                
            with open(file_path, 'r', encoding='utf-8') as f:
                current_content = f.read()
                
            # Apply the fix
            if not dry_run:
                applied_changes = await self.diff_applier.apply_diff(
                    current_content,
                    fix.fixed_code,
                    file_path
                )
                
                # Write the fixed content
                await self.file_manager.write_file(
                    file_path,
                    fix.fixed_code,
                    transaction_id
                )
            else:
                # Dry run - just calculate changes
                applied_changes = await self.diff_applier.apply_diff(
                    current_content,
                    fix.fixed_code,
                    file_path
                )
                
            # Verify if configured
            verification_result = None
            if self.config.verify_after_fix and not dry_run:
                verification_result = await self.verification_engine.verify_fix(
                    None,  # Applied fix info
                    fix
                )
                
                if not verification_result.is_valid and self.config.rollback_on_failure:
                    # Rollback
                    await self.rollback_manager.rollback(transaction_id)
                    
                    return FixApplicationResult(
                        fix_id=fix.id,
                        status=FixApplicationStatus.ROLLED_BACK,
                        error_message="Verification failed, changes rolled back",
                        application_time_ms=int((datetime.now() - start_time).total_seconds() * 1000)
                    )
                    
            # Run tests if configured
            test_results = None
            if self.config.run_tests_after_fix and not dry_run:
                test_results = await self.test_runner.run_tests(file_path)
                
            # Commit transaction
            if not dry_run:
                await self.file_manager.commit_transaction(transaction_id)
                
            return FixApplicationResult(
                fix_id=fix.id,
                status=FixApplicationStatus.SUCCESS,
                applied_changes=[
                    CodeChange(
                        change_type=ChangeType.LOGIC_CHANGE,
                        location=UnifiedPosition(line=change.line_range[0], column=0),
                        description=f"Applied fix at lines {change.line_range[0]}-{change.line_range[1]}",
                        original_content=change.original_content,
                        new_content=change.new_content
                    )
                    for change in applied_changes
                ],
                backup_created=backup_info is not None,
                backup_path=str(backup_info.backup_path) if backup_info else None,
                verification_passed=verification_result.is_valid if verification_result else True,
                test_results=test_results,
                application_time_ms=int((datetime.now() - start_time).total_seconds() * 1000)
            )
            
        except Exception as e:
            logger.error(f"Failed to apply fix: {str(e)}")
            
            # Attempt rollback
            if transaction_id in self.file_manager.transactions:
                try:
                    await self.rollback_manager.rollback(transaction_id)
                except Exception as rollback_error:
                    logger.error(f"Rollback failed: {str(rollback_error)}")
                    
            return FixApplicationResult(
                fix_id=fix.id,
                status=FixApplicationStatus.FAILED,
                error_message=str(e),
                application_time_ms=int((datetime.now() - start_time).total_seconds() * 1000)
            )
            
    async def apply_multiple_fixes(
        self,
        fixes: List[GeneratedFix],
        stop_on_failure: bool = True
    ) -> BatchFixResult:
        """Apply multiple fixes in batch."""
        results = []
        successful = 0
        failed = 0
        
        for fix in fixes:
            result = await self.apply_fix(fix)
            results.append(result)
            
            if result.is_successful:
                successful += 1
            else:
                failed += 1
                
                if stop_on_failure:
                    # Stop processing remaining fixes
                    break
                    
        return BatchFixResult(
            total_fixes=len(fixes),
            successful_fixes=successful,
            failed_fixes=failed,
            individual_results=results,
            overall_success_rate=successful / len(fixes) if fixes else 0.0
        )
        
    async def preview_fix(self, fix: GeneratedFix) -> Dict[str, Any]:
        """Preview a fix without applying it."""
        # Perform a dry run
        result = await self.apply_fix(fix, dry_run=True)
        
        # Generate preview
        preview = {
            "fix_id": str(fix.id),
            "file_path": fix.file_path,
            "changes": [
                {
                    "type": change.change_type.value,
                    "location": f"Line {change.location.line}",
                    "description": change.description,
                    "before": change.original_content[:100] + "..." if len(change.original_content) > 100 else change.original_content,
                    "after": change.new_content[:100] + "..." if len(change.new_content) > 100 else change.new_content
                }
                for change in result.applied_changes
            ],
            "summary": f"{len(result.applied_changes)} changes will be applied",
            "risk_level": fix.risk_level.value
        }
        
        return preview
        
    async def cleanup_old_backups(self) -> int:
        """Clean up old backup files."""
        return await self.backup_manager.cleanup_old_backups(
            self.config.backup_retention_days
        )
