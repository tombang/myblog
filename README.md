# 哈基米的博客

这是一个基于 Hugo 和 PaperMod 的个人博客，使用 GitHub Actions 构建并部署到 GitHub Pages。

## 本地运行

```bash
hugo server -D
```

打开 <http://localhost:1313/> 预览。

## 创建文章

```bash
hugo new content/posts/my-post.md
```

编辑文章并将 front matter 中的 `draft` 改为 `false` 后，推送到 `main` 即可触发部署。

## 本地构建

```bash
hugo --gc --minify --environment production
```

生成目录 `public/` 不提交到 Git，由 GitHub Actions 负责上传部署产物。
