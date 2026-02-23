'use client';

import { useState, useCallback, useMemo } from 'react';

// --- Types ---
interface Blob {
    hash: string;
    content: string;
    label: string;
    emoji: string;
}

interface TreeEntry {
    name: string;
    blobHash: string;
}

interface Tree {
    hash: string;
    entries: TreeEntry[];
}

interface Commit {
    hash: string;
    message: string;
    treeHash: string;
    parentHash: string | null;
    timestamp: string;
    author: string;
}

interface GameState {
    blobs: Record<string, Blob>;
    trees: Record<string, Tree>;
    commits: Record<string, Commit>;
    branches: Record<string, string>; // branch name -> commit hash
    head: string; // current branch name
    workingInventory: { slot: string; item: string; emoji: string }[];
    phase: number;
    stepInPhase: number;
    message: string | null;
    showCelebration: boolean;
}

// --- Helpers ---
let hashCounter = 0;
function makeHash(prefix: string): string {
    hashCounter++;
    const chars = '0123456789abcdef';
    let h = '';
    for (let i = 0; i < 6; i++) h += chars[(hashCounter * 7 + i * 13) % 16];
    return `${prefix}${h}`;
}

function shortHash(h: string): string {
    return h.slice(0, 7);
}

// --- Phase definitions ---
const PHASES = [
    {
        title: '🎮 Chapter 1: Blob — 道具数据',
        subtitle: 'Every item in the game world is stored as a Blob',
        description: '在游戏世界里，每个道具都是一段纯数据。一把铁剑、一瓶药水——它们不知道自己在谁的背包里，只保存自己的属性。Git 的 Blob 也完全一样：纯内容，不关心文件名或路径。',
    },
    {
        title: '🗺️ Chapter 2: Tree — 背包与地图',
        subtitle: 'Trees record where every item goes',
        description: 'Tree 记录了"铁剑放在背包第一格"、"药水放在第二格"。在 Git 里，Tree 就是你的文件夹目录结构——它把 Blob 组织成有意义的位置关系。',
    },
    {
        title: '💾 Chapter 3: Commit — 存档封面',
        subtitle: 'A Commit is the save record, not the save itself',
        description: 'Commit 不是存档内容本身，而是存档的"封面/记录单"。它告诉你：谁存的、什么时候存的、附带了什么备注，然后指向一个 Tree 来表示当时世界的完整状态。',
    },
    {
        title: '🏷️ Chapter 4: Branch — 存档槽位',
        subtitle: 'A Branch is just a lightweight label',
        description: 'Branch 就是游戏里的"存档槽1"或者"自动存档"。它本身没有实质内容，只是一个轻量级标签，贴在某个 Commit 上。切换分支 = 读档，瞬间切换！',
    },
    {
        title: '🔀 Chapter 5: Merge — 合并平行宇宙',
        subtitle: 'Combine two diverged timelines',
        description: '主角在两条平行支线分别打通了不同任务，获得了不同道具。Merge 就是把两条时间线的存档完美融合在一起！',
    },
];

const INITIAL_ITEMS = [
    { slot: 'Slot 1', item: '铁剑 ATK+10', emoji: '⚔️' },
    { slot: 'Slot 2', item: '生命药水 HP+50', emoji: '🧪' },
    { slot: 'Slot 3', item: '皮甲 DEF+5', emoji: '🛡️' },
];

// --- Component ---
export function GitSaveGame() {
    const [state, setState] = useState<GameState>({
        blobs: {},
        trees: {},
        commits: {},
        branches: {},
        head: 'main',
        workingInventory: [...INITIAL_ITEMS],
        phase: 0,
        stepInPhase: 0,
        message: null,
        showCelebration: false,
    });

    const currentPhase = PHASES[state.phase] || PHASES[PHASES.length - 1];

    // --- Phase 0: Create Blobs ---
    const handleCreateBlobs = useCallback(() => {
        const newBlobs: Record<string, Blob> = {};
        state.workingInventory.forEach((item) => {
            const hash = makeHash('b');
            newBlobs[hash] = { hash, content: item.item, label: item.slot, emoji: item.emoji };
        });
        setState((s) => ({
            ...s,
            blobs: { ...s.blobs, ...newBlobs },
            stepInPhase: 1,
            message: `✨ Created ${Object.keys(newBlobs).length} Blobs! Each item is now a unique data object with its own hash.`,
        }));
    }, [state.workingInventory]);

    // --- Phase 1: Create Tree ---
    const handleCreateTree = useCallback(() => {
        const blobEntries = Object.values(state.blobs);
        if (blobEntries.length === 0) return;
        const treeHash = makeHash('t');
        const entries: TreeEntry[] = blobEntries.map((b) => ({
            name: b.label,
            blobHash: b.hash,
        }));
        const tree: Tree = { hash: treeHash, entries };
        setState((s) => ({
            ...s,
            trees: { ...s.trees, [treeHash]: tree },
            stepInPhase: 1,
            message: `🗂️ Tree created! It maps each slot to its Blob hash — this IS your folder structure.`,
        }));
    }, [state.blobs]);

    // --- Phase 2: Create Commit ---
    const handleCreateCommit = useCallback(() => {
        const treeHashes = Object.keys(state.trees);
        const latestTree = treeHashes[treeHashes.length - 1];
        if (!latestTree) return;
        const commitHash = makeHash('c');
        const parentCommits = Object.keys(state.commits);
        const parent = parentCommits.length > 0 ? parentCommits[parentCommits.length - 1] : null;
        const commit: Commit = {
            hash: commitHash,
            message: parent ? '获得新道具' : '初始存档 — 冒险开始！',
            treeHash: latestTree,
            parentHash: parent,
            timestamp: new Date().toLocaleTimeString(),
            author: '勇者',
        };
        setState((s) => ({
            ...s,
            commits: { ...s.commits, [commitHash]: commit },
            stepInPhase: 1,
            message: `📋 Commit created! The save record points to Tree ${shortHash(latestTree)} and knows its parent.`,
        }));
    }, [state.trees, state.commits]);

    // --- Phase 3: Create/Move Branch ---
    const handleCreateBranch = useCallback(() => {
        const commitHashes = Object.keys(state.commits);
        const latest = commitHashes[commitHashes.length - 1];
        if (!latest) return;
        const isFirst = Object.keys(state.branches).length === 0;
        const branchName = isFirst ? 'main' : 'side-quest';
        setState((s) => ({
            ...s,
            branches: { ...s.branches, [branchName]: latest },
            head: branchName,
            stepInPhase: 1,
            message: isFirst
                ? `🏷️ Branch "main" now points to Commit ${shortHash(latest)}. It's just a label!`
                : `🌿 New branch "side-quest" created! You're now on a parallel timeline.`,
        }));
    }, [state.commits, state.branches]);

    // --- Phase 4: Merge ---
    const handleMerge = useCallback(() => {
        // Create a merge commit
        const commitHashes = Object.keys(state.commits);
        const latest = commitHashes[commitHashes.length - 1];
        // Create a merged tree with all items
        const mergedItems = [
            ...INITIAL_ITEMS,
            { slot: 'Slot 4', item: '火焰法杖 MATK+25', emoji: '🔥' },
            { slot: 'Slot 5', item: '暗影斗篷 EVA+15', emoji: '🧥' },
        ];
        const newBlobs: Record<string, Blob> = {};
        mergedItems.forEach((item) => {
            const hash = makeHash('b');
            newBlobs[hash] = { hash, content: item.item, label: item.slot, emoji: item.emoji };
        });
        const treeHash = makeHash('t');
        const entries: TreeEntry[] = Object.values(newBlobs).map((b) => ({
            name: b.label,
            blobHash: b.hash,
        }));
        const tree: Tree = { hash: treeHash, entries };
        const commitHash = makeHash('c');
        const commit: Commit = {
            hash: commitHash,
            message: 'Merge: 支线任务完成，合并所有道具！',
            treeHash: treeHash,
            parentHash: latest || null,
            timestamp: new Date().toLocaleTimeString(),
            author: '勇者',
        };
        setState((s) => ({
            ...s,
            blobs: { ...s.blobs, ...newBlobs },
            trees: { ...s.trees, [treeHash]: tree },
            commits: { ...s.commits, [commitHash]: commit },
            branches: { ...s.branches, main: commitHash },
            head: 'main',
            workingInventory: mergedItems,
            stepInPhase: 1,
            showCelebration: true,
            message: `🎉 Merge complete! Both timelines combined — you now have ALL the loot!`,
        }));
    }, [state.commits]);

    // --- Advance to next phase ---
    const handleNextPhase = useCallback(() => {
        setState((s) => ({
            ...s,
            phase: Math.min(s.phase + 1, PHASES.length - 1),
            stepInPhase: 0,
            message: null,
            showCelebration: false,
        }));
    }, []);

    // Add side-quest item when entering phase 4
    const handleAddSideQuestItem = useCallback(() => {
        const newItem = { slot: 'Slot 4', item: '火焰法杖 MATK+25', emoji: '🔥' };
        const blobHash = makeHash('b');
        const newBlob: Blob = { hash: blobHash, content: newItem.item, label: newItem.slot, emoji: newItem.emoji };
        const treeHash = makeHash('t');
        const allBlobs = { ...state.blobs, [blobHash]: newBlob };
        const entries: TreeEntry[] = Object.values(allBlobs).map((b) => ({
            name: b.label,
            blobHash: b.hash,
        }));
        const tree: Tree = { hash: treeHash, entries };
        const commitHash = makeHash('c');
        const commitHashes = Object.keys(state.commits);
        const parent = commitHashes[commitHashes.length - 1] || null;
        const commit: Commit = {
            hash: commitHash,
            message: '支线任务：获得火焰法杖',
            treeHash,
            parentHash: parent,
            timestamp: new Date().toLocaleTimeString(),
            author: '勇者',
        };
        setState((s) => ({
            ...s,
            blobs: allBlobs,
            trees: { ...s.trees, [treeHash]: tree },
            commits: { ...s.commits, [commitHash]: commit },
            branches: { ...s.branches, 'side-quest': commitHash },
            workingInventory: [...s.workingInventory, newItem],
            stepInPhase: 1,
            message: `🔥 Side quest complete! You got a Fire Staff on the "side-quest" branch. Now merge it back!`,
        }));
    }, [state.blobs, state.commits]);

    // --- Action for current phase ---
    const phaseAction = useMemo(() => {
        const actions = [
            { label: '⚔️ Store Items as Blobs', handler: handleCreateBlobs, done: state.stepInPhase > 0 },
            { label: '🗂️ Organize into a Tree', handler: handleCreateTree, done: state.stepInPhase > 0 },
            { label: '💾 Create Save Record (Commit)', handler: handleCreateCommit, done: state.stepInPhase > 0 },
            { label: '🏷️ Attach Branch Label', handler: handleCreateBranch, done: state.stepInPhase > 0 },
            { label: state.stepInPhase === 0 ? '🌿 Complete Side Quest' : '🔀 Merge Timelines', handler: state.stepInPhase === 0 ? handleAddSideQuestItem : handleMerge, done: state.showCelebration },
        ];
        return actions[state.phase] || actions[actions.length - 1];
    }, [state.phase, state.stepInPhase, state.showCelebration, handleCreateBlobs, handleCreateTree, handleCreateCommit, handleCreateBranch, handleAddSideQuestItem, handleMerge]);

    const isLastPhase = state.phase === PHASES.length - 1;
    const canAdvance = state.stepInPhase > 0 && !isLastPhase;

    // --- Render ---
    return (
        <div className="my-8 rounded-xl border-2 border-brand-orange/30 bg-gradient-to-br from-[hsl(var(--card))] to-[hsl(var(--muted))] overflow-hidden" style={{ fontFamily: "'Poppins', sans-serif" }}>
            {/* Header */}
            <div className="bg-gradient-to-r from-orange-500/20 via-purple-500/10 to-blue-500/20 dark:from-orange-500/10 dark:via-purple-500/5 dark:to-blue-500/10 px-4 sm:px-6 py-4 border-b border-border">
                <div className="flex items-center gap-2 mb-1">
                    <span className="text-lg">🎮</span>
                    <h3 className="text-base sm:text-lg font-bold text-foreground m-0">Git Save Game — 用游戏存档理解 Git</h3>
                </div>
                <p className="text-xs text-muted-foreground m-0">Interactive tutorial • Click through each chapter</p>
            </div>

            {/* Phase indicator */}
            <div className="px-4 sm:px-6 pt-4">
                <div className="flex gap-1 mb-4">
                    {PHASES.map((_, i) => (
                        <div
                            key={i}
                            className={`h-1.5 flex-1 rounded-full transition-all duration-500 ${i < state.phase ? 'bg-green-500' : i === state.phase ? 'bg-brand-orange animate-pulse' : 'bg-border'
                                }`}
                        />
                    ))}
                </div>

                {/* Current phase info */}
                <div className="mb-4">
                    <h4 className="text-sm sm:text-base font-bold text-foreground m-0">{currentPhase.title}</h4>
                    <p className="text-xs text-brand-orange m-0 mt-0.5">{currentPhase.subtitle}</p>
                    <p className="text-xs text-muted-foreground m-0 mt-2 leading-relaxed">{currentPhase.description}</p>
                </div>
            </div>

            {/* Game area */}
            <div className="px-4 sm:px-6 pb-4 grid grid-cols-1 lg:grid-cols-2 gap-4">
                {/* Left: Working inventory */}
                <div className="rounded-lg border border-border bg-[hsl(var(--card))] p-3">
                    <h5 className="text-xs font-semibold text-muted-foreground uppercase tracking-wider m-0 mb-2">
                        🎒 Working Inventory (工作区)
                    </h5>
                    <div className="space-y-1.5">
                        {state.workingInventory.map((item, i) => (
                            <div
                                key={i}
                                className="flex items-center gap-2 px-2.5 py-1.5 rounded-md bg-[hsl(var(--muted))] text-xs transition-all duration-300"
                                style={{ animationDelay: `${i * 100}ms` }}
                            >
                                <span className="text-base">{item.emoji}</span>
                                <span className="font-medium text-foreground">{item.slot}:</span>
                                <span className="text-muted-foreground">{item.item}</span>
                            </div>
                        ))}
                    </div>
                </div>

                {/* Right: Git object store */}
                <div className="rounded-lg border border-border bg-[hsl(var(--card))] p-3">
                    <h5 className="text-xs font-semibold text-muted-foreground uppercase tracking-wider m-0 mb-2">
                        📦 Git Object Store (.git/objects)
                    </h5>
                    <div className="space-y-2 max-h-64 overflow-y-auto" style={{ scrollbarWidth: 'thin' }}>
                        {/* Branches */}
                        {Object.entries(state.branches).length > 0 && (
                            <div>
                                <span className="text-[10px] font-bold text-purple-500 uppercase">Branches</span>
                                {Object.entries(state.branches).map(([name, commitHash]) => (
                                    <div key={name} className={`flex items-center gap-1.5 px-2 py-1 rounded text-[11px] mt-0.5 ${state.head === name ? 'bg-purple-500/10 border border-purple-500/30' : 'bg-[hsl(var(--muted))]'}`}>
                                        <span>{state.head === name ? '👉' : '🏷️'}</span>
                                        <span className="font-bold text-purple-500">{name}</span>
                                        <span className="text-muted-foreground">→ {shortHash(commitHash)}</span>
                                    </div>
                                ))}
                            </div>
                        )}

                        {/* Commits */}
                        {Object.values(state.commits).length > 0 && (
                            <div>
                                <span className="text-[10px] font-bold text-blue-500 uppercase">Commits</span>
                                {Object.values(state.commits).reverse().map((c) => (
                                    <div key={c.hash} className="px-2 py-1 rounded bg-[hsl(var(--muted))] mt-0.5 text-[11px]">
                                        <div className="flex items-center gap-1.5">
                                            <span>💾</span>
                                            <code className="font-mono text-blue-500">{shortHash(c.hash)}</code>
                                            <span className="text-muted-foreground truncate">{c.message}</span>
                                        </div>
                                        <div className="text-[10px] text-muted-foreground ml-5">
                                            tree: {shortHash(c.treeHash)} | parent: {c.parentHash ? shortHash(c.parentHash) : 'none'} | {c.author} @ {c.timestamp}
                                        </div>
                                    </div>
                                ))}
                            </div>
                        )}

                        {/* Trees */}
                        {Object.values(state.trees).length > 0 && (
                            <div>
                                <span className="text-[10px] font-bold text-green-500 uppercase">Trees</span>
                                {Object.values(state.trees).reverse().map((t) => (
                                    <div key={t.hash} className="px-2 py-1 rounded bg-[hsl(var(--muted))] mt-0.5 text-[11px]">
                                        <div className="flex items-center gap-1.5">
                                            <span>🗂️</span>
                                            <code className="font-mono text-green-500">{shortHash(t.hash)}</code>
                                        </div>
                                        {t.entries.map((e, i) => (
                                            <div key={i} className="text-[10px] text-muted-foreground ml-5">
                                                {e.name} → blob:{shortHash(e.blobHash)}
                                            </div>
                                        ))}
                                    </div>
                                ))}
                            </div>
                        )}

                        {/* Blobs */}
                        {Object.values(state.blobs).length > 0 && (
                            <div>
                                <span className="text-[10px] font-bold text-orange-500 uppercase">Blobs</span>
                                {Object.values(state.blobs).map((b) => (
                                    <div key={b.hash} className="flex items-center gap-1.5 px-2 py-1 rounded bg-[hsl(var(--muted))] mt-0.5 text-[11px]">
                                        <span>{b.emoji}</span>
                                        <code className="font-mono text-orange-500">{shortHash(b.hash)}</code>
                                        <span className="text-muted-foreground">{b.content}</span>
                                    </div>
                                ))}
                            </div>
                        )}

                        {Object.keys(state.blobs).length === 0 && (
                            <p className="text-xs text-muted-foreground italic m-0">Empty — start storing items!</p>
                        )}
                    </div>
                </div>
            </div>

            {/* Message */}
            {state.message && (
                <div className="mx-4 sm:mx-6 mb-3 px-3 py-2 rounded-lg bg-brand-orange/10 border border-brand-orange/20 text-xs text-foreground leading-relaxed animate-[fadeIn_0.3s_ease]">
                    {state.message}
                </div>
            )}

            {/* Action buttons */}
            <div className="px-4 sm:px-6 pb-4 flex flex-wrap gap-2">
                {!phaseAction.done && (
                    <button
                        onClick={phaseAction.handler}
                        className="px-4 py-2 rounded-lg bg-brand-orange text-white text-xs font-semibold hover:bg-brand-orange/90 transition-all duration-200 hover:scale-[1.02] active:scale-95 shadow-md"
                    >
                        {phaseAction.label}
                    </button>
                )}
                {canAdvance && (
                    <button
                        onClick={handleNextPhase}
                        className="px-4 py-2 rounded-lg border border-border bg-[hsl(var(--card))] text-foreground text-xs font-semibold hover:bg-[hsl(var(--muted))] transition-all duration-200"
                    >
                        Next Chapter →
                    </button>
                )}
                {state.showCelebration && (
                    <div className="flex items-center gap-2 px-3 py-2 rounded-lg bg-green-500/10 border border-green-500/30 text-xs text-green-600 dark:text-green-400 font-semibold animate-[fadeIn_0.5s_ease]">
                        🎉 Tutorial Complete! You now understand Git internals!
                    </div>
                )}
            </div>

            {/* Visual diagram */}
            {state.phase >= 2 && Object.keys(state.commits).length > 0 && (
                <div className="px-4 sm:px-6 pb-4">
                    <div className="rounded-lg border border-border bg-[hsl(var(--card))] p-3">
                        <h5 className="text-xs font-semibold text-muted-foreground uppercase tracking-wider m-0 mb-2">
                            🔗 The Full Picture
                        </h5>
                        <div className="flex flex-wrap items-center gap-1.5 text-[11px]">
                            {Object.entries(state.branches).map(([name]) => (
                                <span key={name} className="px-1.5 py-0.5 rounded bg-purple-500/15 text-purple-500 font-bold">
                                    {name}
                                </span>
                            ))}
                            <span className="text-muted-foreground">→</span>
                            <span className="px-1.5 py-0.5 rounded bg-blue-500/15 text-blue-500 font-mono">
                                Commit
                            </span>
                            <span className="text-muted-foreground">→</span>
                            <span className="px-1.5 py-0.5 rounded bg-green-500/15 text-green-500 font-mono">
                                Tree
                            </span>
                            <span className="text-muted-foreground">→</span>
                            <span className="px-1.5 py-0.5 rounded bg-orange-500/15 text-orange-500 font-mono">
                                Blobs
                            </span>
                        </div>
                        <p className="text-[10px] text-muted-foreground m-0 mt-1.5">
                            Branch (标签) → Commit (封面) → Tree (目录) → Blob (数据) — 这就是 Git 的全部秘密！
                        </p>
                    </div>
                </div>
            )}
        </div>
    );
}
